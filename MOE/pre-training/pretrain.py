import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from uer.opts import *


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =========================
    # Path options
    # =========================
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--config_path", type=str,
                        default="models/bert/base_config.json",
                        help="Config file of model hyper-parameters.")
    parser.add_argument("--tensorboard_log_dir", type=str, default="runs/moe_pretrain",
                        help="TensorBoard log directory.")
    parser.add_argument("--tensorboard_param_steps", type=int, default=10,
                        help="Specific steps to write parameter and gradient norms.")
    parser.add_argument("--tensorboard_histogram_steps", type=int, default=200,
                        help="Specific steps to write selected parameter histograms.")
    parser.add_argument("--disable_tensorboard_graph", action="store_true",
                        help="Disable TensorBoard graph tracing.")
    parser.add_argument("--training_state_path", type=str, default=None,
                        help="Path of the resumable training state checkpoint.")
    parser.add_argument("--resume_training_state_path", type=str, default=None,
                        help="Path of the training state checkpoint to resume from.")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Resume automatically from training_state_path if it exists.")
    parser.add_argument("--state_save_steps", type=int, default=50,
                        help="Specific steps to save resumable training state.")

    # =========================
    # Training options
    # =========================
    # >>> MOE MOD: 缩小默认规模，方便先跑通
    parser.add_argument("--total_steps", type=int, default=10,
                        help="Total training steps (small for MoE debug).")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=1,
                        help="Specific steps to print prompt.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size (small for MoE debug).")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    # <<< MOE MOD

    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--labels_num", type=int, required=False,
                        help="Number of prediction labels.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout value.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # =========================
    # Preprocess options
    # =========================
    parser.add_argument("--tokenizer",
                        choices=["bert", "char", "space"],
                        default="bert",
                        help="Tokenizer type.")

    # =========================
    # Model options
    # =========================
    model_opts(parser)

    parser.add_argument("--tgt_embedding",
                        choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"],
                        default="word_pos_seg",
                        help="Target embedding type.")
    parser.add_argument("--decoder",
                        choices=["transformer"],
                        default="transformer",
                        help="Decoder type.")
    parser.add_argument("--pooling",
                        choices=["mean", "max", "first", "last"],
                        default="first",
                        help="Pooling type.")
    parser.add_argument("--target",
                        choices=["bert", "lm", "mlm", "bilm", "albert",
                                 "seq2seq", "t5", "cls", "prefixlm"],
                        default="bert",
                        help="Training target.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")

    # =========================
    # Masking options
    # =========================
    parser.add_argument("--whole_word_masking", action="store_true",
                        help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true",
                        help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Geo prob for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max span length.")

    # =========================
    # Optimizer options
    # =========================
    optimization_opts(parser)

    # =========================
    # GPU / Distributed options
    # =========================
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs).")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int,
                        help="Ranks of each process.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345",
                        type=str, help="Master IP.")
    parser.add_argument("--backend",
                        choices=["nccl", "gloo"],
                        default="nccl",
                        type=str,
                        help="Distributed backend.")

    args = parser.parse_args()

    # =========================
    # Sanity checks
    # =========================
    if args.target == "cls":
        assert args.labels_num is not None, \
            "Cls target needs labels_num."

    # =========================
    # Load hyperparameters
    # =========================
    if args.config_path:
        args = load_hyperparam(args)

    # =========================
    # Device / distributed setup
    # =========================
    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size
        assert ranks_num <= torch.cuda.device_count()
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode.")
    elif args.world_size == 1 and ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count()
        args.dist_train = False
        args.single_gpu = True
        print(f"Using GPU {args.gpu_id}.")
    else:
        assert ranks_num == 0
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode.")

    # =========================
    # Train
    # =========================
    trainer.train_and_validate(args)


if __name__ == "__main__":
    main()
