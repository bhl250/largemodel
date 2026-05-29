"""
MoE-compatible classifier fine-tuning for ET-BERT/UER.
"""

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from uer.encoders import *
from uer.layers import *
from uer.model_saver import save_model
from uer.opts import finetune_opts
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.seed import set_seed


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.moe_aux_weight = getattr(args, "moe_aux_weight", 0.01)

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        enc_out = self.encoder(emb, seg)

        if isinstance(enc_out, tuple):
            output, aux_loss = enc_out[0], enc_out[1]
        else:
            output, aux_loss = enc_out, None

        if self.pooling == "mean":
            pooled = torch.mean(output, dim=1)
        elif self.pooling == "max":
            pooled = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            pooled = output[:, -1, :]
        else:
            pooled = output[:, 0, :]

        pooled = torch.tanh(self.output_layer_1(pooled))
        logits = self.output_layer_2(pooled)

        if tgt is None:
            return None, logits

        cls_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
        if aux_loss is not None:
            return cls_loss + self.moe_aux_weight * aux_loss, logits
        return cls_loss, logits


def count_labels_num(path):
    labels = set()
    columns = {}
    with open(path, mode="r", encoding="utf-8") as reader:
        for line_id, line in enumerate(reader):
            parts = line.rstrip("\n").split("\t")
            if line_id == 0:
                columns = {name: i for i, name in enumerate(parts)}
                continue
            if not parts or len(parts) <= columns["label"]:
                continue
            labels.add(int(parts[columns["label"]]))
    return len(labels)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is None:
        for name, param in model.named_parameters():
            if param.dim() > 1 and "gamma" not in name and "beta" not in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
        return

    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print("Loaded pretrained model:", args.pretrained_model_path)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))


def build_optimizer(args, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=args.learning_rate)

    warmup_steps = int(train_steps * args.warmup)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def encode_text(args, text_a):
    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
    seg = [1] * len(src)

    src = src[:args.seq_length]
    seg = seg[:args.seq_length]

    while len(src) < args.seq_length:
        src.append(0)
        seg.append(0)

    return src, seg


def read_dataset(args, path, with_label=True):
    src, tgt, seg = [], [], []
    columns = {}

    with open(path, mode="r", encoding="utf-8") as reader:
        for line_id, line in enumerate(reader):
            parts = line.rstrip("\n").split("\t")
            if line_id == 0:
                columns = {name: i for i, name in enumerate(parts)}
                if "text_a" not in columns:
                    raise ValueError("{} must contain text_a column.".format(path))
                if with_label and "label" not in columns:
                    raise ValueError("{} must contain label column.".format(path))
                continue
            if len(parts) <= columns["text_a"]:
                continue

            encoded_src, encoded_seg = encode_text(args, parts[columns["text_a"]])
            src.append(encoded_src)
            seg.append(encoded_seg)
            if with_label:
                tgt.append(int(parts[columns["label"]]))

    src = torch.LongTensor(src)
    seg = torch.LongTensor(seg)
    if with_label:
        tgt = torch.LongTensor(tgt)
    else:
        tgt = None
    return src, tgt, seg


def batch_indices(size, batch_size, shuffle=False):
    indices = list(range(size))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start:start + batch_size]


def get_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def evaluate(args, model, dataset, split_name, print_confusion_matrix=False):
    src, tgt, seg = dataset
    model.eval()

    correct = 0
    total = src.size(0)
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for ids in batch_indices(total, args.batch_size, shuffle=False):
            src_batch = src[ids].to(args.device)
            tgt_batch = tgt[ids].to(args.device)
            seg_batch = seg[ids].to(args.device)

            loss, logits = model(src_batch, tgt_batch, seg_batch)
            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            batches += 1
            correct += torch.sum(pred == tgt_batch).item()
            for predicted, gold in zip(pred.cpu(), tgt_batch.cpu()):
                confusion[predicted.item(), gold.item()] += 1

    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, batches)
    precision, recall, f1 = macro_scores(confusion)

    print(
        "{} loss {:.4f} acc {:.4f} macro_p {:.4f} macro_r {:.4f} macro_f1 {:.4f} ({}/{})".format(
            split_name, avg_loss, accuracy, precision, recall, f1, correct, total
        )
    )

    if print_confusion_matrix:
        print("{} confusion matrix:".format(split_name))
        print(confusion)

    return {
        "loss": avg_loss,
        "acc": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion": confusion,
    }


def macro_scores(confusion):
    eps = 1e-9
    f1_sum, precision_sum, recall_sum = 0.0, 0.0, 0.0
    labels_num = confusion.size(0)
    for label in range(labels_num):
        tp = confusion[label, label].item()
        predicted = confusion[label, :].sum().item()
        gold = confusion[:, label].sum().item()
        precision = tp / (predicted + eps)
        recall = tp / (gold + eps)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    return precision_sum / labels_num, recall_sum / labels_num, f1_sum / labels_num


def setup_writer(args):
    if not args.tensorboard_log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard is required. Install it with: pip install tensorboard") from exc
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    writer.add_text("run/pretrained_model_path", str(args.pretrained_model_path), 0)
    writer.add_text("run/output_model_path", str(args.output_model_path), 0)
    writer.add_text("run/train_path", str(args.train_path), 0)
    writer.add_text("run/dev_path", str(args.dev_path), 0)
    writer.add_text("run/test_path", str(args.test_path), 0)
    return writer


def write_eval_scalars(writer, metrics, split_name, step):
    if writer is None:
        return
    for key, value in metrics.items():
        if key == "confusion":
            continue
        writer.add_scalar("{}/{}".format(split_name, key), value, step)


def train(args, model, train_dataset, dev_dataset, test_dataset):
    train_src, train_tgt, train_seg = train_dataset
    train_size = train_src.size(0)
    train_steps = ((train_size + args.batch_size - 1) // args.batch_size) * args.epochs_num
    optimizer, scheduler = build_optimizer(args, model, train_steps)
    writer = setup_writer(args)

    best_dev_acc = -1.0
    best_dev_f1 = -1.0
    global_step = 0

    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs_num)
    print("Train instances:", train_size)
    print("Total optimization steps:", train_steps)

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        total_loss = 0.0
        report_loss = 0.0

        for ids in batch_indices(train_size, args.batch_size, shuffle=True):
            src_batch = train_src[ids].to(args.device)
            tgt_batch = train_tgt[ids].to(args.device)
            seg_batch = train_seg[ids].to(args.device)

            optimizer.zero_grad()
            loss, _ = model(src_batch, tgt_batch, seg_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            total_loss += loss_value
            report_loss += loss_value
            global_step += 1

            if writer is not None:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % args.report_steps == 0:
                print(
                    "epoch {} step {}/{} avg_loss {:.4f} lr {:.8f}".format(
                        epoch,
                        global_step,
                        train_steps,
                        report_loss / args.report_steps,
                        optimizer.param_groups[0]["lr"],
                    )
                )
                report_loss = 0.0

        print("Epoch {} train avg_loss {:.4f}".format(epoch, total_loss / max(1, train_size // args.batch_size)))

        dev_metrics = evaluate(args, model, dev_dataset, "dev")
        write_eval_scalars(writer, dev_metrics, "dev", global_step)

        should_save = dev_metrics["acc"] > best_dev_acc
        if dev_metrics["acc"] == best_dev_acc and dev_metrics["macro_f1"] > best_dev_f1:
            should_save = True

        if should_save:
            best_dev_acc = dev_metrics["acc"]
            best_dev_f1 = dev_metrics["macro_f1"]
            save_model(model, args.output_model_path)
            print("Saved best model to {}".format(args.output_model_path))

    print("Best dev acc {:.4f}, best dev macro_f1 {:.4f}".format(best_dev_acc, best_dev_f1))

    if test_dataset is not None:
        print("Evaluating final in-memory model on test set:")
        test_metrics = evaluate(args, model, test_dataset, "test", print_confusion_matrix=args.print_confusion_matrix)
        write_eval_scalars(writer, test_metrics, "test", global_step)

    if writer is not None:
        writer.flush()
        writer.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer.")
    parser.add_argument("--moe_aux_weight", type=float, default=0.01,
                        help="Weight for MoE auxiliary loss.")
    parser.add_argument("--tensorboard_log_dir", type=str, default=None,
                        help="TensorBoard log directory.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm.")
    parser.add_argument("--print_confusion_matrix", action="store_true",
                        help="Print full confusion matrix after test evaluation.")

    args = parser.parse_args()
    args = load_hyperparam(args)

    set_seed(args.seed)
    args.labels_num = count_labels_num(args.train_path)
    print("Labels num:", args.labels_num)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = Classifier(args)
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Using DataParallel.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    print("Loading train dataset:", args.train_path)
    train_dataset = read_dataset(args, args.train_path, with_label=True)
    print("Loading dev dataset:", args.dev_path)
    dev_dataset = read_dataset(args, args.dev_path, with_label=True)
    test_dataset = None
    if args.test_path:
        print("Loading test dataset:", args.test_path)
        test_dataset = read_dataset(args, args.test_path, with_label=True)

    train(args, model, train_dataset, dev_dataset, test_dataset)


if __name__ == "__main__":
    main()
