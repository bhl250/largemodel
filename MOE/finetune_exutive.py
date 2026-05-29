import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
GPU_ID = "1"

PRETRAINED_MODEL = PROJECT_ROOT / "models" / "moe_pre-trained_model.bin-100000"
VOCAB_PATH = PROJECT_ROOT / "models" / "encryptd_vocab.txt"
CONFIG_PATH = PROJECT_ROOT / "models" / "bert" / "base_config.json"

DATA_DIR = PROJECT_ROOT / "datasets" / "cstnet-tls1.3" / "packet"
TRAIN_PATH = DATA_DIR / "train_dataset.tsv"
DEV_PATH = DATA_DIR / "valid_dataset.tsv"
TEST_PATH = DATA_DIR / "test_dataset.tsv"

OUTPUT_MODEL = PROJECT_ROOT / "models" / "moe_finetuned_cstnet_packet.bin"
TENSORBOARD_DIR = PROJECT_ROOT / "runs" / "moe_finetune_cstnet_packet"
BACKGROUND_LOG = TENSORBOARD_DIR / "background_finetune.log"
PID_PATH = PROJECT_ROOT / "models" / "moe_finetune.pid"

TENSORBOARD_PORT = "6007"


def require_file(path):
    if not path.exists():
        raise FileNotFoundError("Required file does not exist: {}".format(path))


def is_process_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def training_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env.setdefault("PYTHONUNBUFFERED", "1")
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def start_tensorboard():
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TENSORBOARD_DIR / "tensorboard_server.log"
    command = [
        PYTHON,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(PROJECT_ROOT / "runs"),
        "--host",
        "0.0.0.0",
        "--port",
        TENSORBOARD_PORT,
    ]
    with open(log_path, "ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(2)
    if process.poll() is None:
        print("TensorBoard started on port {}.".format(TENSORBOARD_PORT))
    else:
        print("TensorBoard may already be running or failed to start. Check: {}".format(log_path))


def finetune_command():
    return [
        PYTHON,
        "fine-tuning/run_classifier.py",
        "--pretrained_model_path",
        str(PRETRAINED_MODEL),
        "--vocab_path",
        str(VOCAB_PATH),
        "--config_path",
        str(CONFIG_PATH),
        "--train_path",
        str(TRAIN_PATH),
        "--dev_path",
        str(DEV_PATH),
        "--test_path",
        str(TEST_PATH),
        "--output_model_path",
        str(OUTPUT_MODEL),
        "--epochs_num",
        "5",
        "--batch_size",
        "32",
        "--seq_length",
        "128",
        "--learning_rate",
        "2e-5",
        "--report_steps",
        "500",
        "--embedding",
        "word_pos_seg",
        "--encoder",
        "transformer",
        "--mask",
        "fully_visible",
        "--moe_experts",
        "4",
        "--moe_top_k",
        "2",
        "--moe_balance_coef",
        "0.01",
        "--moe_z_loss_coef",
        "0.001",
        "--moe_aux_weight",
        "0.01",
        "--tensorboard_log_dir",
        str(TENSORBOARD_DIR),
    ]


def print_help(pid):
    print("\nBackground fine-tuning process started.")
    print("PID:")
    print("  {}".format(pid))
    print("Status:")
    print("  ps -p {}".format(pid))
    print("Logs:")
    print("  tail -f {}".format(BACKGROUND_LOG))
    print("Stop:")
    print("  kill {}".format(pid))
    print("\nOutputs:")
    print("  fine-tuned model: {}".format(OUTPUT_MODEL))
    print("  pid file:         {}".format(PID_PATH))
    print("  tensorboard dir:  {}".format(TENSORBOARD_DIR))
    print("  tensorboard URL:  http://127.0.0.1:{}".format(TENSORBOARD_PORT))


def run_finetune_foreground():
    os.chdir(PROJECT_ROOT)
    for path in [PRETRAINED_MODEL, VOCAB_PATH, CONFIG_PATH, TRAIN_PATH, DEV_PATH, TEST_PATH]:
        require_file(path)

    start_tensorboard()
    command = finetune_command()
    print("\n$ " + " ".join(str(item) for item in command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=training_env()).returncode


def start_background_finetune():
    os.chdir(PROJECT_ROOT)
    for path in [PRETRAINED_MODEL, VOCAB_PATH, CONFIG_PATH, TRAIN_PATH, DEV_PATH, TEST_PATH]:
        require_file(path)

    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
        except ValueError:
            old_pid = None
        if old_pid is not None and is_process_running(old_pid):
            print_help(old_pid)
            print("\nFine-tuning already appears to be running; not starting a duplicate process.")
            return 0

    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, str(PROJECT_ROOT / "finetune_exutive.py"), "--run-finetune"]
    with open(BACKGROUND_LOG, "ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=training_env(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_PATH.write_text(str(process.pid) + "\n")
    time.sleep(2)
    if process.poll() is not None:
        print("Fine-tuning process exited immediately with code {}.".format(process.returncode))
        print("Check log: {}".format(BACKGROUND_LOG))
        return process.returncode

    print_help(process.pid)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Start ET-BERT MoE CSTNET packet fine-tuning in the background."
    )
    parser.add_argument("--run-finetune", action="store_true",
                        help="Internal mode used by the detached background process.")
    args = parser.parse_args()

    if args.run_finetune:
        return run_finetune_foreground()
    return start_background_finetune()


if __name__ == "__main__":
    raise SystemExit(main())
