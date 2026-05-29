import os
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
SAVED_RUNS_DIR = PROJECT_ROOT / "saved_runs"

TRAIN_STATE = MODELS_DIR / "moe_pretrain_latest_state.pt"
TRAIN_PID = MODELS_DIR / "moe_pretrain.pid"
TRAIN_TENSORBOARD_DIR = RUNS_DIR / "moe_pretrain"
SMOKE_TENSORBOARD_DIR = RUNS_DIR / "moe_smoke_test"


def is_process_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def assert_no_active_training():
    if not TRAIN_PID.exists():
        return

    try:
        pid = int(TRAIN_PID.read_text().strip())
    except ValueError:
        return

    if is_process_running(pid):
        raise RuntimeError(
            "Training process still appears to be running: pid {}. "
            "Stop it first or wait for it to finish.".format(pid)
        )


def unique_archive_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = SAVED_RUNS_DIR / ("moe_pretrain_" + timestamp)
    index = 1
    while archive_dir.exists():
        archive_dir = SAVED_RUNS_DIR / ("moe_pretrain_{}_{}".format(timestamp, index))
        index += 1
    return archive_dir


def move_path(src, dst, moved):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    moved.append((src, dst))


def main():
    os.chdir(PROJECT_ROOT)
    assert_no_active_training()

    archive_dir = unique_archive_dir()
    archive_models = archive_dir / "models"
    archive_runs = archive_dir / "runs"
    moved = []

    for path in sorted(MODELS_DIR.glob("moe_pre-trained_model.bin-*")):
        move_path(path, archive_models / path.name, moved)

    for path in sorted(MODELS_DIR.glob("moe_smoke_test.bin-*")):
        move_path(path, archive_models / path.name, moved)

    for path in [
        TRAIN_STATE,
        TRAIN_PID,
        MODELS_DIR / "moe_smoke_test.state.pt",
    ]:
        move_path(path, archive_models / path.name, moved)

    move_path(TRAIN_TENSORBOARD_DIR, archive_runs / TRAIN_TENSORBOARD_DIR.name, moved)
    move_path(SMOKE_TENSORBOARD_DIR, archive_runs / SMOKE_TENSORBOARD_DIR.name, moved)

    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / "MANIFEST.txt"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        manifest.write("Archived ET-BERT MoE run\n")
        manifest.write("archive_dir: {}\n\n".format(archive_dir))
        if not moved:
            manifest.write("No previous run artifacts were found.\n")
        else:
            manifest.write("Moved artifacts:\n")
            for src, dst in moved:
                manifest.write("- {} -> {}\n".format(src, dst))

    print("Previous run artifacts saved to:")
    print("  {}".format(archive_dir))
    print("Manifest:")
    print("  {}".format(manifest_path))
    if not moved:
        print("No previous run artifacts were found.")
    else:
        print("Moved {} artifact(s).".format(len(moved)))
    print("\nYou can now start a fresh formal run with:")
    print("  python exutive.py")


if __name__ == "__main__":
    raise SystemExit(main())
