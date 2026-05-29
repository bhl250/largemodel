import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
import tqdm


def split_aux_loss(loss_info, expected_items):
    """
    Target outputs already contain tensors such as loss, correct, denominator.
    MoE auxiliary loss is appended as the final item by uer.models.Model.
    """
    if not isinstance(loss_info, tuple):
        loss_info = (loss_info,)

    aux_loss = None
    if len(loss_info) > expected_items:
        aux_loss = loss_info[-1]
        loss_info = loss_info[:-1]

    return loss_info, aux_loss


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def save_training_state(model, optimizer, scheduler, step, path):
    state = {
        "step": step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(state, path)


def load_training_state(model, optimizer, scheduler, path):
    state = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", 0))


def sanitize_tag(name):
    return name.replace(".", "/")


def parameter_group_name(name):
    if "embedding" in name:
        return "embedding"
    if "self_attn" in name or "context_attn" in name:
        return "attention"
    if "feed_forward.moe.gate" in name:
        return "moe_router"
    if "feed_forward.moe.experts" in name:
        return "moe_experts"
    if "feed_forward.ff_after" in name:
        return "moe_ff_after"
    if "layer_norm" in name or "layer_norm" in name or name.endswith(".gamma") or name.endswith(".beta"):
        return "layer_norm"
    if "target" in name or "output_layer" in name or "nsp" in name or "sop" in name or "mlm" in name:
        return "target_head"
    return "other"


def tensor_norm(tensor):
    return torch.norm(tensor.detach().float()).item()


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece")
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        args.vocab = {sp_model.IdToPiece(i): i for i
                      in range(sp_model.GetPieceSize())}
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        if args.target == "seq2seq":
            tgt_sp_model = spm.SentencePieceProcessor()
            tgt_sp_model.Load(args.tgt_spm_model_path)
            args.tgt_vocab = {tgt_sp_model.IdToPiece(i): i for i
                              in range(tgt_sp_model.GetPieceSize())}
    else:
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        args.vocab = args.tokenizer.vocab
        if args.target == "seq2seq":
            tgt_vocab = Vocab()
            tgt_vocab.load(args.tgt_vocab_path)
            args.tgt_vocab = tgt_vocab.w2i

    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)


class Trainer(object):
    def __init__(self, args):
        self.current_step = getattr(args, "resume_step", 1)
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.state_save_steps = args.state_save_steps

        self.output_model_path = args.output_model_path
        self.training_state_path = args.training_state_path or args.output_model_path + ".training_state.pt"

        self.start_time = time.time()
        self.total_loss = 0.0
        self.last_scalars = {}
        self.writer = None
        self.graph_written = False
        self.last_completed_step = self.current_step - 1

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size

    def forward_propagation(self, batch, model):
        raise NotImplementedError

    def report_and_reset_stats(self):
        raise NotImplementedError

    def setup_tensorboard(self, args, rank):
        if self.dist_train and rank != 0:
            return
        if not args.tensorboard_log_dir:
            raise ValueError("tensorboard_log_dir is required for this training script.")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("TensorBoard is required. Install it with: pip install tensorboard") from exc
        self.writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
        self.writer.add_text("run/output_model_path", args.output_model_path, 0)
        self.writer.add_text("run/training_state_path", self.training_state_path, 0)
        self.writer.add_text("run/moe", "experts={}, top_k={}, balance_coef={}, z_loss_coef={}".format(
            args.moe_experts, args.moe_top_k, args.moe_balance_coef, args.moe_z_loss_coef
        ), 0)

    def write_graph_once(self, args, model, batch):
        if self.graph_written or args.disable_tensorboard_graph or self.writer is None:
            return
        try:
            src = batch[0]
            seg = batch[-1]
            if args.target in ["bert", "albert"]:
                model_input = (src, (batch[1], batch[2]), seg)
            elif args.target == "bilm":
                model_input = (src, (batch[1], batch[2]), seg)
            elif args.target in ["seq2seq", "t5"]:
                model_input = (src, (batch[1], batch[2], src), seg)
            else:
                model_input = (src, batch[1], seg)
            self.writer.add_graph(unwrap_model(model), model_input)
            self.writer.add_text("graph/status", "TensorBoard graph traced successfully.", self.current_step)
        except Exception as exc:
            self.writer.add_text("graph/status", "TensorBoard graph tracing failed: {}".format(repr(exc)),
                                 self.current_step)
        self.graph_written = True

    def log_tensorboard_scalars(self, optimizer):
        if self.writer is None:
            return
        step = self.current_step
        for key, value in self.last_scalars.items():
            self.writer.add_scalar(key, value, step)
        if optimizer.param_groups:
            self.writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], step)

    def log_tensorboard_parameters(self, args, model):
        if self.writer is None or self.current_step % args.tensorboard_param_steps != 0:
            return
        group_param_norms = {}
        group_grad_norms = {}
        for name, param in unwrap_model(model).named_parameters():
            group = parameter_group_name(name)
            param_norm = tensor_norm(param.data)
            group_param_norms[group] = group_param_norms.get(group, 0.0) + param_norm ** 2
            self.writer.add_scalar("param_norm/" + sanitize_tag(name), param_norm, self.current_step)
            if param.grad is not None:
                grad_norm = tensor_norm(param.grad)
                group_grad_norms[group] = group_grad_norms.get(group, 0.0) + grad_norm ** 2
                self.writer.add_scalar("grad_norm/" + sanitize_tag(name), grad_norm, self.current_step)

        for group, norm_sq in group_param_norms.items():
            self.writer.add_scalar("param_group_norm/" + group, norm_sq ** 0.5, self.current_step)
        for group, norm_sq in group_grad_norms.items():
            self.writer.add_scalar("grad_group_norm/" + group, norm_sq ** 0.5, self.current_step)

    def log_tensorboard_histograms(self, args, model):
        if self.writer is None or self.current_step % args.tensorboard_histogram_steps != 0:
            return
        for name, param in unwrap_model(model).named_parameters():
            if ("embedding.word_embedding" in name or
                    "self_attn.final_linear.weight" in name or
                    "feed_forward.moe.gate.to_gates.weight" in name or
                    "feed_forward.moe.experts.experts.0" in name or
                    "target" in name):
                self.writer.add_histogram("hist_params/" + sanitize_tag(name), param.detach().float().cpu(),
                                          self.current_step)
                if param.grad is not None:
                    self.writer.add_histogram("hist_grads/" + sanitize_tag(name), param.grad.detach().float().cpu(),
                                              self.current_step)

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)
        self.setup_tensorboard(args, rank)

        try:
            for _ in tqdm.tqdm(range(self.total_steps)):
                if self.current_step > self.total_steps:
                    break

                batch = list(next(loader_iter))
                self.seq_length = batch[0].size(1)

                if gpu_id is not None:
                    for i in range(len(batch)):
                        batch[i] = batch[i].cuda(gpu_id)

                loss = self.forward_propagation(batch, model)
                self.write_graph_once(args, model, batch)

                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if not self.dist_train or (self.dist_train and rank == 0):
                    self.log_tensorboard_scalars(optimizer)
                    self.log_tensorboard_parameters(args, model)
                    self.log_tensorboard_histograms(args, model)

                if self.current_step % self.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if self.current_step % self.report_steps == 0 and \
                        (not self.dist_train or (self.dist_train and rank == 0)):
                    self.report_and_reset_stats()
                    self.start_time = time.time()

                if self.current_step % self.save_checkpoint_steps == 0 and \
                        (not self.dist_train or (self.dist_train and rank == 0)):
                    save_model(model, self.output_model_path + "-" + str(self.current_step))

                if self.current_step % self.state_save_steps == 0 and \
                        (not self.dist_train or (self.dist_train and rank == 0)):
                    save_training_state(model, optimizer, scheduler, self.current_step, self.training_state_path)

                self.last_completed_step = self.current_step
                self.current_step += 1
        finally:
            if not self.dist_train or (self.dist_train and rank == 0):
                save_training_state(model, optimizer, scheduler, self.last_completed_step, self.training_state_path)
                if self.writer is not None:
                    self.writer.flush()
                    self.writer.close()


class MlmTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)

        full, aux_loss = split_aux_loss(loss_info, 3)
        loss, correct, denominator = full[:3]

        total_loss = loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        aux_value = aux_loss.item() if aux_loss is not None else 0.0
        self.last_scalars = {
            "loss/total": total_loss.item(),
            "loss/mlm": loss.item(),
            "loss/moe_aux": aux_value,
            "accuracy/mlm": correct.item() / denominator.item()
        }

        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print(f"| {self.current_step:8d}/{self.total_steps:8d} "
              f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s "
              f"| loss {self.total_loss / self.report_steps:7.2f} "
              f"| acc {self.total_correct / self.total_denominator:.4f}")

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class NspTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_sp, seg = batch
        loss_info = model(src, (tgt_mlm, tgt_sp), seg)

        full, aux_loss = split_aux_loss(loss_info, 5)
        loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = full[:5]

        total_loss = loss_sp
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        aux_value = aux_loss.item() if aux_loss is not None else 0.0
        self.last_scalars = {
            "loss/total": total_loss.item(),
            "loss/mlm": loss_mlm.item(),
            "loss/nsp": loss_sp.item(),
            "loss/moe_aux": aux_value,
            "accuracy/nsp": correct_sp.item() / src.size(0)
        }

        self.total_loss += total_loss.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)

        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print(f"| {self.current_step:8d}/{self.total_steps:8d} "
              f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s "
              f"| loss {self.total_loss / self.report_steps:7.2f} "
              f"| loss_sp: {self.total_loss_sp / self.report_steps:.4f} "
              f"| acc_sp: {self.total_correct_sp / self.total_instances:.4f}")

        self.total_loss = 0.0
        self.total_loss_sp = 0.0
        self.total_denominator = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0


class BertTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.total_loss_mlm = 0.0
        self.total_loss_sp = 0.0
        self.total_correct_mlm = 0.0
        self.total_correct_sp = 0.0
        self.total_denominator = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_sp, seg = batch
        loss_info = model(src, (tgt_mlm, tgt_sp), seg)

        full, aux_loss = split_aux_loss(loss_info, 5)
        loss_mlm, loss_sp, correct_mlm, correct_sp, denominator = full[:5]

        total_loss = loss_mlm + loss_sp
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        aux_value = aux_loss.item() if aux_loss is not None else 0.0
        self.last_scalars = {
            "loss/total": total_loss.item(),
            "loss/mlm": loss_mlm.item(),
            "loss/nsp": loss_sp.item(),
            "loss/moe_aux": aux_value,
            "accuracy/mlm": correct_mlm.item() / denominator.item(),
            "accuracy/nsp": correct_sp.item() / src.size(0)
        }

        self.total_loss += total_loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)

        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print(f"| {self.current_step:8d}/{self.total_steps:8d} "
              f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s "
              f"| loss {self.total_loss / self.report_steps:7.2f} "
              f"| loss_mlm: {self.total_loss_mlm / self.report_steps:.4f} "
              f"| loss_sp: {self.total_loss_sp / self.report_steps:.4f} "
              f"| acc_mlm: {self.total_correct_mlm / self.total_denominator:.4f} "
              f"| acc_sp: {self.total_correct_sp / self.total_instances:.4f}")

        self.total_loss = 0.0
        self.total_loss_mlm = 0.0
        self.total_loss_sp = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0


class AlbertTrainer(BertTrainer):
    pass


class LmTrainer(MlmTrainer):
    pass


class BilmTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_forward, tgt_backward, seg = batch
        loss_info = model(src, (tgt_forward, tgt_backward), seg)

        full, aux_loss = split_aux_loss(loss_info, 5)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = full[:5]

        total_loss = loss_forward + loss_backward
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        aux_value = aux_loss.item() if aux_loss is not None else 0.0
        self.last_scalars = {
            "loss/total": total_loss.item(),
            "loss/forward": loss_forward.item(),
            "loss/backward": loss_backward.item(),
            "loss/moe_aux": aux_value,
            "accuracy/forward": correct_forward.item() / denominator.item(),
            "accuracy/backward": correct_backward.item() / denominator.item()
        }

        self.total_loss += total_loss.item()
        self.total_loss_forward += loss_forward.item()
        self.total_loss_backward += loss_backward.item()
        self.total_correct_forward += correct_forward.item()
        self.total_correct_backward += correct_backward.item()
        self.total_denominator += denominator.item()

        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print(f"| {self.current_step:8d}/{self.total_steps:8d} "
              f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s "
              f"| loss {self.total_loss / self.report_steps:7.2f} "
              f"| loss_forward {self.total_loss_forward / self.report_steps:.4f} "
              f"| loss_backward {self.total_loss_backward / self.report_steps:.4f} "
              f"| acc_forward: {self.total_correct_forward / self.total_denominator:.4f} "
              f"| acc_backward: {self.total_correct_backward / self.total_denominator:.4f}")

        self.total_loss = 0.0
        self.total_loss_forward = 0.0
        self.total_loss_backward = 0.0
        self.total_correct_forward = 0.0
        self.total_correct_backward = 0.0
        self.total_denominator = 0.0


class ClsTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.total_correct = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)

        full, aux_loss = split_aux_loss(loss_info, 2)
        loss, correct = full[:2]

        total_loss = loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        aux_value = aux_loss.item() if aux_loss is not None else 0.0
        self.last_scalars = {
            "loss/total": total_loss.item(),
            "loss/classification": loss.item(),
            "loss/moe_aux": aux_value,
            "accuracy/classification": correct.item() / src.size(0)
        }

        self.total_loss += total_loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)

        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print(f"| {self.current_step:8d}/{self.total_steps:8d} "
              f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s "
              f"| loss {self.total_loss / self.report_steps:7.2f} "
              f"| acc: {self.total_correct / self.total_instances:.4f}")

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class Seq2seqTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_in, tgt_out, seg = batch
        loss_info = model(src, (tgt_in, tgt_out, src), seg)

        full, aux_loss = split_aux_loss(loss_info, 3)
        loss, correct, denominator = full[:3]

        total_loss = loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        aux_value = aux_loss.item() if aux_loss is not None else 0.0
        self.last_scalars = {
            "loss/total": total_loss.item(),
            "loss/seq2seq": loss.item(),
            "loss/moe_aux": aux_value,
            "accuracy/seq2seq": correct.item() / denominator.item()
        }

        self.total_loss += total_loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        return total_loss / self.accumulation_steps

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print(f"| {self.current_step:8d}/{self.total_steps:8d} "
              f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s "
              f"| loss {self.total_loss / self.report_steps:7.2f} "
              f"| acc: {self.total_correct / self.total_denominator:.4f}")

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class T5Trainer(Seq2seqTrainer):
    pass


class PrefixlmTrainer(MlmTrainer):
    pass


str2trainer = {
    "bert": BertTrainer,
    "mlm": MlmTrainer,
    "lm": LmTrainer,
    "albert": AlbertTrainer,
    "bilm": BilmTrainer,
    "cls": ClsTrainer,
    "seq2seq": Seq2seqTrainer,
    "t5": T5Trainer,
    "nsp": NspTrainer
}


def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size,
                                                   True)
    else:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup, args.total_steps)

    if args.training_state_path is None:
        args.training_state_path = args.output_model_path + ".training_state.pt"

    resume_path = args.resume_training_state_path
    if resume_path is None and args.auto_resume and os.path.exists(args.training_state_path):
        resume_path = args.training_state_path

    if resume_path is not None:
        resumed_step = load_training_state(model, optimizer, scheduler, resume_path)
        args.resume_step = resumed_step + 1
        print("Resumed training state from {} at step {}.".format(resume_path, resumed_step))
    else:
        args.resume_step = 1

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if args.dist_train:
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print(f"Worker {rank} is training ...")
    else:
        print("Worker is training ...")

    trainer = str2trainer[args.target](args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
