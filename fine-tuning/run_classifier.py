"""
This script provides an example to wrap UER-py for classification.
(MoE compatible version: run-only, not for real training)
"""

import random
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np


# =========================
# MoE helper
# =========================
def unpack_loss(loss_info):
    """
    Compatible with:
    - loss
    - (loss, aux_loss)
    - (loss, aux_loss, ...)
    """
    if isinstance(loss_info, tuple):
        loss = loss_info[0]
        aux_loss = None
        if len(loss_info) > 1 and torch.is_tensor(loss_info[1]):
            aux_loss = loss_info[1]
        return loss, aux_loss
    else:
        return loss_info, None


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.labels_num = args.labels_num
        self.pooling = args.pooling

        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        # MoE auxiliary loss weight（只为能跑）
        self.moe_aux_weight = getattr(args, "moe_aux_weight", 0.01)

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)

        # ===== Encoder (MoE compatible) =====
        enc_out = self.encoder(emb, seg)

        aux_loss = None
        if isinstance(enc_out, tuple):
            output = enc_out[0]
            if len(enc_out) > 1:
                aux_loss = enc_out[1]
        else:
            output = enc_out

        # ===== Pooling =====
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

        cls_loss = nn.NLLLoss()(
            nn.LogSoftmax(dim=-1)(logits),
            tgt.view(-1)
        )

        # ===== Add MoE auxiliary loss =====
        if aux_loss is not None:
            total_loss = cls_loss + self.moe_aux_weight * aux_loss
        else:
            total_loss = cls_loss

        return total_loss, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        model.load_state_dict(
            torch.load(args.pretrained_model_path, map_location="cpu"),
            strict=False
        )
    else:
        for n, p in model.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)


def build_optimizer(args, model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0
    )
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg):
    total = src.size(0)
    for i in range(0, total, batch_size):
        yield (
            src[i:i + batch_size],
            tgt[i:i + batch_size],
            seg[i:i + batch_size]
        )


def read_dataset(args, path, max_samples=64):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue

            if len(dataset) >= max_samples:
                break

            line = line.strip().split("\t")
            tgt = int(line[columns["label"]])
            text_a = line[columns["text_a"]]

            src = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
            )
            seg = [1] * len(src)

            src = src[:args.seq_length]
            seg = seg[:args.seq_length]

            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)

            dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)

        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)

        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch

        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()

        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--moe_aux_weight", type=float, default=0.01,
                        help="Weight for MoE auxiliary loss.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # ===== 快速测试配置 =====
    args.batch_size = 2
    args.epochs_num = 1
    args.seq_length = min(args.seq_length, 64)
    args.report_steps = 10

    set_seed(42)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path, max_samples=64)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)
    print("Training for quick test (not real training)")

    optimizer, scheduler = build_optimizer(args, model)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss = 0.0

    print("Start training for quick test.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        print(f"Epoch {epoch} completed. Loss: {loss.item():.4f}")

    print("Quick test completed successfully!")

    # 简单的评估
    if args.dev_path:
        devset = read_dataset(args, args.dev_path, max_samples=32)
        acc, _ = evaluate(args, devset)
        print(f"Quick evaluation on dev set: {acc:.4f}")

    print("Finetuning script run OK. (MoE compatible quick test)")


if __name__ == "__main__":
    main()