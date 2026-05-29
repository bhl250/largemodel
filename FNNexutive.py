import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
GPU_ID = "1"

DATASET_PATH = PROJECT_ROOT / "dataset.pt"
VOCAB_PATH = PROJECT_ROOT / "models" / "encryptd_vocab.txt"
CONFIG_PATH = PROJECT_ROOT / "models" / "bert" / "base_config.json"

FNN_CODE_DIR = PROJECT_ROOT / "FNN"
PRETRAIN_SCRIPT = FNN_CODE_DIR / "pre-training" / "pretrain.py"

SMOKE_OUTPUT = PROJECT_ROOT / "models" / "FNN_smoke_test.bin"
SMOKE_STATE = PROJECT_ROOT / "models" / "FNN_smoke_test.state.pt"
SMOKE_TENSORBOARD_DIR = PROJECT_ROOT / "runs" / "FNN_smoke_test"

TRAIN_OUTPUT = PROJECT_ROOT / "models" / "FNN_pre-trained_model.bin"
TRAIN_STATE = PROJECT_ROOT / "models" / "FNN_pretrain_latest_state.pt"
TRAIN_PID = PROJECT_ROOT / "models" / "FNN_pretrain.pid"
TRAIN_TENSORBOARD_DIR = PROJECT_ROOT / "runs" / "FNN_pretrain"
BACKGROUND_LOG = TRAIN_TENSORBOARD_DIR / "background_train.log"

TRAIN_TOTAL_STEPS = 100000
TRAIN_BATCH_SIZE = 8
TRAIN_SEQ_LENGTH = 128
TRAIN_SAVE_CHECKPOINT_STEPS = 5000
TRAIN_STATE_SAVE_STEPS = 100
TRAIN_REPORT_STEPS = 50
TENSORBOARD_PARAM_STEPS = 100
TENSORBOARD_HISTOGRAM_STEPS = 1000
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
        env["PYTHONPATH"] = str(FNN_CODE_DIR) + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = str(FNN_CODE_DIR)
    return env


def run(command, env=None):
    print("\n$ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def start_tensorboard():
    TRAIN_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TRAIN_TENSORBOARD_DIR / "tensorboard_server.log"
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


def base_pretrain_command(output_path, tensorboard_dir, state_path):
    return [
        PYTHON,
        str(PRETRAIN_SCRIPT),
        "--dataset_path",
        str(DATASET_PATH),
        "--vocab_path",
        str(VOCAB_PATH),
        "--config_path",
        str(CONFIG_PATH),
        "--output_model_path",
        str(output_path),
        "--target",
        "bert",
        "--embedding",
        "word_pos_seg",
        "--encoder",
        "transformer",
        "--mask",
        "fully_visible",
        "--gpu_ranks",
        "0",
        "--seq_length",
        str(TRAIN_SEQ_LENGTH),
        "--instances_buffer_size",
        "25600",
        "--learning_rate",
        "2e-5",
        "--tensorboard_log_dir",
        str(tensorboard_dir),
        "--tensorboard_param_steps",
        str(TENSORBOARD_PARAM_STEPS),
        "--tensorboard_histogram_steps",
        str(TENSORBOARD_HISTOGRAM_STEPS),
        "--training_state_path",
        str(state_path),
    ]


def smoke_test(env):
    if TRAIN_STATE.exists():
        print("Existing FNN training state found; skip smoke test and resume training.")
        return

    print("Start FNN smoke test: 10 steps on physical GPU {}.".format(GPU_ID))
    shutil.rmtree(SMOKE_TENSORBOARD_DIR, ignore_errors=True)
    for path in [SMOKE_STATE, PROJECT_ROOT / "models" / "FNN_smoke_test.bin-10"]:
        if path.exists():
            path.unlink()

    command = base_pretrain_command(SMOKE_OUTPUT, SMOKE_TENSORBOARD_DIR, SMOKE_STATE)
    command.extend([
        "--batch_size",
        "1",
        "--total_steps",
        "10",
        "--save_checkpoint_steps",
        "10",
        "--state_save_steps",
        "10",
        "--report_steps",
        "1",
    ])
    run(command, env=env)

    print("FNN smoke test succeeded.")
    for path in [SMOKE_STATE, PROJECT_ROOT / "models" / "FNN_smoke_test.bin-10"]:
        if path.exists():
            path.unlink()


def train(env):
    print("Start FNN main pretraining on physical GPU {}.".format(GPU_ID))
    print("If interrupted, run this script again to resume from {}.".format(TRAIN_STATE))

    command = base_pretrain_command(TRAIN_OUTPUT, TRAIN_TENSORBOARD_DIR, TRAIN_STATE)
    command.extend([
        "--batch_size",
        str(TRAIN_BATCH_SIZE),
        "--total_steps",
        str(TRAIN_TOTAL_STEPS),
        "--save_checkpoint_steps",
        str(TRAIN_SAVE_CHECKPOINT_STEPS),
        "--state_save_steps",
        str(TRAIN_STATE_SAVE_STEPS),
        "--report_steps",
        str(TRAIN_REPORT_STEPS),
        "--auto_resume",
    ])
    run(command, env=env)

    print("FNN main pretraining finished.")
    print("Latest FNN training state: {}".format(TRAIN_STATE))
    print("FNN model checkpoint prefix: {}".format(TRAIN_OUTPUT))
    print("TensorBoard log dir: {}".format(TRAIN_TENSORBOARD_DIR))


def run_training_foreground():
    os.chdir(PROJECT_ROOT)
    for path in [DATASET_PATH, VOCAB_PATH, CONFIG_PATH, PRETRAIN_SCRIPT]:
        require_file(path)

    start_tensorboard()
    env = training_env()
    try:
        smoke_test(env)
        train(env)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run this script to resume from the latest saved FNN state.")
        return 130
    except subprocess.CalledProcessError as exc:
        print("\nCommand failed with exit code {}.".format(exc.returncode))
        print("If FNN main training had started, re-run this script to resume.")
        return exc.returncode
    return 0


def print_background_help(pid):
    print("\nBackground FNN pretraining process started.")
    print("PID:")
    print("  {}".format(pid))
    print("Status:")
    print("  ps -p {}".format(pid))
    print("Logs:")
    print("  tail -f {}".format(BACKGROUND_LOG))
    print("Stop:")
    print("  kill {}".format(pid))
    print("\nOutputs:")
    print("  final checkpoint: {}-{}".format(TRAIN_OUTPUT, TRAIN_TOTAL_STEPS))
    print("  resumable state:  {}".format(TRAIN_STATE))
    print("  pid file:         {}".format(TRAIN_PID))
    print("  tensorboard dir:  {}".format(TRAIN_TENSORBOARD_DIR))
    print("  tensorboard URL:  http://127.0.0.1:{}".format(TENSORBOARD_PORT))


def start_background_training():
    os.chdir(PROJECT_ROOT)
    for path in [DATASET_PATH, VOCAB_PATH, CONFIG_PATH, PRETRAIN_SCRIPT]:
        require_file(path)

    if TRAIN_PID.exists():
        try:
            old_pid = int(TRAIN_PID.read_text().strip())
        except ValueError:
            old_pid = None
        if old_pid is not None and is_process_running(old_pid):
            print_background_help(old_pid)
            print("\nFNN pretraining already appears to be running; not starting a duplicate.")
            return 0

    TRAIN_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, str(PROJECT_ROOT / "FNNexutive.py"), "--run-training"]
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

    TRAIN_PID.write_text(str(process.pid) + "\n")
    time.sleep(2)
    if process.poll() is not None:
        print("FNN pretraining process exited immediately with code {}.".format(process.returncode))
        print("Check log: {}".format(BACKGROUND_LOG))
        return process.returncode

    print_background_help(process.pid)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Start FNN ET-BERT pretraining in the background."
    )
    parser.add_argument("--run-training", action="store_true",
                        help="Internal mode used by the detached background process.")
    args = parser.parse_args()

    if args.run_training:
        return run_training_foreground()
    return start_background_training()


if __name__ == "__main__":
    raise SystemExit(main())
