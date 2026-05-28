import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

GPU_ID = "1"
PYTHON = sys.executable

DATASET_PATH = PROJECT_ROOT / "dataset.pt"
VOCAB_PATH = PROJECT_ROOT / "models" / "encryptd_vocab.txt"
CONFIG_PATH = PROJECT_ROOT / "models" / "bert" / "base_config.json"

SMOKE_OUTPUT = PROJECT_ROOT / "models" / "moe_smoke_test.bin"
SMOKE_STATE = PROJECT_ROOT / "models" / "moe_smoke_test.state.pt"
SMOKE_TENSORBOARD_DIR = PROJECT_ROOT / "runs" / "moe_smoke_test"

TRAIN_OUTPUT = PROJECT_ROOT / "models" / "moe_pre-trained_model.bin"
TRAIN_STATE = PROJECT_ROOT / "models" / "moe_pretrain_latest_state.pt"
TRAIN_TENSORBOARD_DIR = PROJECT_ROOT / "runs" / "moe_pretrain"

TENSORBOARD_PORT = "6007"
SERVICE_NAME = "etbert-moe-pretrain"
SERVICE_DESCRIPTION = "ET-BERT MoE pretraining"


def run_capture(command):
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run(command, env=None):
    print("\n$ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError("Required file does not exist: {}".format(path))


def require_imports():
    import torch
    import st_moe_pytorch  # noqa: F401
    import tensorboard  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    print("torch:", torch.__version__)
    print("cuda devices visible:", torch.cuda.device_count())


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
        print("Log dir: {}".format(PROJECT_ROOT / "runs"))
        print("Server log: {}".format(log_path))
    else:
        print("TensorBoard did not stay running. Training will still write event files.")
        print("Check: {}".format(log_path))


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


def service_paths():
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return "system", Path("/etc/systemd/system") / (SERVICE_NAME + ".service")

    user_dir = Path.home() / ".config" / "systemd" / "user"
    return "user", user_dir / (SERVICE_NAME + ".service")


def systemctl_command(scope, *args):
    command = ["systemctl"]
    if scope == "user":
        command.append("--user")
    command.extend(args)
    return command


def write_service_file(scope, service_path):
    service_path.parent.mkdir(parents=True, exist_ok=True)
    install_target = "multi-user.target"
    if scope == "user":
        install_target = "default.target"

    service_text = """[Unit]
Description={description}
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory={working_directory}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH={pythonpath}
Environment=CUDA_VISIBLE_DEVICES={gpu_id}
ExecStart={python} {script} --run-training
Restart=on-failure
RestartSec=30

[Install]
WantedBy={install_target}
""".format(
        description=SERVICE_DESCRIPTION,
        working_directory=PROJECT_ROOT,
        pythonpath=PROJECT_ROOT,
        gpu_id=GPU_ID,
        python=PYTHON,
        script=PROJECT_ROOT / "exutive.py",
        install_target=install_target,
    )
    service_path.write_text(service_text)


def print_service_help(scope):
    service = SERVICE_NAME + ".service"
    prefix = "systemctl"
    if scope == "user":
        prefix = "systemctl --user"
        journal_prefix = "journalctl --user"
    else:
        journal_prefix = "journalctl"

    print("\nSystemd service started: {}".format(service))
    print("Training runs outside your SSH session.")
    print("Status:")
    print("  {} status {}".format(prefix, service))
    print("Logs:")
    print("  {} -u {} -f".format(journal_prefix, service))
    print("Stop:")
    print("  {} stop {}".format(prefix, service))
    print("\nOutputs:")
    print("  final checkpoint: {}-1000".format(TRAIN_OUTPUT))
    print("  resumable state:  {}".format(TRAIN_STATE))
    print("  tensorboard dir:  {}".format(TRAIN_TENSORBOARD_DIR))
    print("  tensorboard URL:  http://127.0.0.1:{}".format(TENSORBOARD_PORT))


def start_systemd_service():
    os.chdir(PROJECT_ROOT)
    require_file(DATASET_PATH)
    require_file(VOCAB_PATH)
    require_file(CONFIG_PATH)

    systemctl_check = run_capture(["systemctl", "--version"])
    if systemctl_check.returncode != 0:
        print(systemctl_check.stdout)
        raise RuntimeError("systemctl is not available on this server.")

    scope, service_path = service_paths()
    write_service_file(scope, service_path)

    commands = [
        systemctl_command(scope, "daemon-reload"),
        systemctl_command(scope, "restart", SERVICE_NAME + ".service"),
    ]

    for command in commands:
        result = run_capture(command)
        if result.returncode != 0:
            print(result.stdout)
            if scope == "user":
                print("User systemd failed. If this server does not support user services, run as root or use sudo.")
            raise RuntimeError("Failed to run: {}".format(" ".join(command)))

    print_service_help(scope)


def base_pretrain_command(output_path, tensorboard_dir, state_path):
    return [
        PYTHON,
        "pre-training/pretrain.py",
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
        "128",
        "--instances_buffer_size",
        "25600",
        "--moe_experts",
        "4",
        "--moe_top_k",
        "2",
        "--moe_balance_coef",
        "0.01",
        "--moe_z_loss_coef",
        "0.001",
        "--learning_rate",
        "2e-5",
        "--tensorboard_log_dir",
        str(tensorboard_dir),
        "--tensorboard_param_steps",
        "10",
        "--tensorboard_histogram_steps",
        "200",
        "--training_state_path",
        str(state_path),
    ]


def smoke_test(env):
    if TRAIN_STATE.exists():
        print("Existing training state found; skip smoke test and resume training.")
        return

    print("Start smoke test: 10 steps on GPU {}.".format(GPU_ID))
    shutil.rmtree(SMOKE_TENSORBOARD_DIR, ignore_errors=True)
    for path in [SMOKE_STATE, PROJECT_ROOT / "models" / "moe_smoke_test.bin-10"]:
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

    print("Smoke test succeeded.")
    for path in [SMOKE_STATE, PROJECT_ROOT / "models" / "moe_smoke_test.bin-10"]:
        if path.exists():
            path.unlink()


def train(env):
    print("Start main training on GPU {}.".format(GPU_ID))
    print("If interrupted, run this script again to resume from {}.".format(TRAIN_STATE))

    command = base_pretrain_command(TRAIN_OUTPUT, TRAIN_TENSORBOARD_DIR, TRAIN_STATE)
    command.extend([
        "--batch_size",
        "8",
        "--total_steps",
        "1000",
        "--save_checkpoint_steps",
        "1000",
        "--state_save_steps",
        "50",
        "--report_steps",
        "10",
        "--auto_resume",
    ])
    run(command, env=env)

    print("Main training finished.")
    print("Latest training state: {}".format(TRAIN_STATE))
    print("Model checkpoint prefix: {}".format(TRAIN_OUTPUT))
    print("TensorBoard log dir: {}".format(TRAIN_TENSORBOARD_DIR))


def run_training_main():
    os.chdir(PROJECT_ROOT)
    require_file(DATASET_PATH)
    require_file(VOCAB_PATH)
    require_file(CONFIG_PATH)
    require_imports()

    start_tensorboard()
    env = training_env()

    try:
        smoke_test(env)
        train(env)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run this script to resume from the latest saved training state.")
        return 130
    except subprocess.CalledProcessError as exc:
        print("\nCommand failed with exit code {}.".format(exc.returncode))
        print("If main training had already started, re-run this script to resume from the latest state.")
        return exc.returncode

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Start ET-BERT MoE pretraining as a systemd service."
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Internal mode used by systemd. Do not pass this manually unless you want foreground training.",
    )
    args = parser.parse_args()

    if args.run_training:
        return run_training_main()

    try:
        start_systemd_service()
    except Exception as exc:
        print("\nFailed to start systemd service: {}".format(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
