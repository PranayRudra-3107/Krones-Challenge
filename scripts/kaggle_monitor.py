import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kaggle_teacher_jobs.json"
DEFAULT_OUT = ROOT / "outputs" / "kaggle_monitor"
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"


STATUS_RE = re.compile(r'has status "([^"]+)"')
PROGRESS_PATTERNS = [
    "Kaggle dependency check done",
    "DATA:",
    "Checkpoint status:",
    "Running P1 folds:",
    "Running P2 folds:",
    "train:",
    "epoch",
    "P2 OOF F1",
    "Kaggle chunk P1 complete",
    "Kaggle chunk P2 complete",
    "Wrote manifest:",
]

EPOCHS_BY_PHASE = {
    "p1": 5,
    "p2": 4,
}
EPOCH_RE = re.compile(r"Ep\s+(\d+)/(\d+).*?EMA F1=([0-9.]+)@t=([0-9.]+)")
FOLD_RE = re.compile(r"\[P([12])\s+\|.*?\]\s+Fold\s+(\d+)")
SKIP_RE = re.compile(r"p([12]) fold (\d+) done - skip")
SWA_RE = re.compile(r"SWA\(.*?\)\s+F1=([0-9.]+)@t=([0-9.]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_kaggle(args: list[str], timeout: int = 120) -> tuple[int, str]:
    if not KAGGLE_BIN.exists():
        return 127, f"Missing Kaggle CLI: {KAGGLE_BIN}"
    result = subprocess.run(
        [str(KAGGLE_BIN), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def load_jobs(config_path: Path) -> list[dict]:
    with config_path.open() as handle:
        payload = json.load(handle)
    jobs = payload.get("jobs", payload)
    if not isinstance(jobs, list):
        raise ValueError(f"Expected a jobs list in {config_path}")
    return jobs


def parse_status(raw: str, returncode: int) -> str:
    if returncode != 0:
        lowered = raw.lower()
        if "wrong kernel slug" in lowered or "permission" in lowered or "cannot access kernel" in lowered:
            return "NOT_FOUND"
        return "STATUS_ERROR"
    match = STATUS_RE.search(raw)
    if not match:
        return "UNKNOWN"
    return match.group(1).replace("KernelWorkerStatus.", "")


def simplify_logs(raw: str, lines: int) -> tuple[str, list[str]]:
    if not raw:
        return "", []
    raw_lines = raw.splitlines()
    tail = "\n".join(raw_lines[-lines:])
    progress = []
    for line in raw_lines:
        if any(pattern in line for pattern in PROGRESS_PATTERNS):
            progress.append(line)
    return tail, progress[-10:]


def iter_log_events(raw: str) -> list[dict]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [item for item in parsed if isinstance(item, dict)]


def log_text_lines(raw: str) -> list[str]:
    events = iter_log_events(raw)
    if not events:
        return raw.splitlines()
    out = []
    for event in events:
        data = str(event.get("data", ""))
        out.extend(line for line in data.splitlines() if line.strip())
    return out


def parse_progress_state(job: dict, raw_logs: str, status: str) -> dict:
    planned_folds = list(job.get("folds", []))
    phase = str(job.get("phase", "")).lower()
    epochs_per_fold = EPOCHS_BY_PHASE.get(phase, 5)
    total_units = max(1, len(planned_folds) * epochs_per_fold)

    current_fold = None
    current_epoch = 0
    current_epoch_total = epochs_per_fold
    completed_folds: set[int] = set()
    last_f1 = None
    last_threshold = None
    last_signal = ""

    for line in log_text_lines(raw_logs):
        fold_match = FOLD_RE.search(line)
        if fold_match:
            current_fold = int(fold_match.group(2))
            current_epoch = 0
            last_signal = f"fold {current_fold} started"

        skip_match = SKIP_RE.search(line)
        if skip_match:
            completed_folds.add(int(skip_match.group(2)))
            last_signal = f"fold {skip_match.group(2)} skipped"

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_epoch_total = int(epoch_match.group(2))
            epochs_per_fold = current_epoch_total
            last_f1 = float(epoch_match.group(3))
            last_threshold = float(epoch_match.group(4))
            last_signal = f"epoch {current_epoch}/{current_epoch_total}"

        swa_match = SWA_RE.search(line)
        if swa_match:
            if current_fold is not None:
                completed_folds.add(current_fold)
            last_f1 = float(swa_match.group(1))
            last_threshold = float(swa_match.group(2))
            last_signal = "SWA complete"

        if f"Kaggle chunk {phase.upper()} complete" in line and planned_folds:
            completed_folds.update(planned_folds)
            current_epoch = epochs_per_fold
            last_signal = f"{phase.upper()} complete"

    if status in {"COMPLETE", "SUCCEEDED"} and planned_folds:
        completed_folds.update(planned_folds)

    completed_units = len(completed_folds & set(planned_folds)) * epochs_per_fold
    if current_fold in planned_folds and current_fold not in completed_folds:
        completed_units += min(current_epoch, epochs_per_fold)
    completed_units = min(completed_units, total_units)
    percent = completed_units / total_units * 100

    return {
        "percent": percent,
        "completed_units": completed_units,
        "total_units": total_units,
        "current_fold": current_fold,
        "current_epoch": current_epoch,
        "current_epoch_total": current_epoch_total,
        "completed_folds": sorted(completed_folds & set(planned_folds)),
        "last_f1": last_f1,
        "last_threshold": last_threshold,
        "last_signal": last_signal,
    }


def progress_bar(percent: float, width: int = 22) -> str:
    filled = int(round(width * max(0.0, min(100.0, percent)) / 100))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def check_job(job: dict, log_lines: int, fetch_logs: bool) -> dict:
    ref = job["ref"]
    status_code, status_raw = run_kaggle(["kernels", "status", ref], timeout=60)
    status = parse_status(status_raw, status_code)

    log_tail = ""
    progress = []
    logs_code = None
    logs_raw = ""
    if fetch_logs and status not in {"NOT_FOUND", "STATUS_ERROR"}:
        logs_code, logs_raw = run_kaggle(["kernels", "logs", ref], timeout=120)
        if logs_code == 0:
            log_tail, progress = simplify_logs(logs_raw, log_lines)
        else:
            log_tail = logs_raw
    progress_state = parse_progress_state(job, logs_raw, status) if logs_code == 0 else {}

    return {
        "checked_at": utc_now(),
        "name": job.get("name", ref),
        "model": job.get("model", ""),
        "phase": job.get("phase", ""),
        "folds": job.get("folds", []),
        "ref": ref,
        "status": status,
        "status_raw": status_raw,
        "logs_returncode": logs_code,
        "progress": progress,
        "progress_state": progress_state,
        "log_tail": log_tail,
        "output_dir": job.get("output_dir", ""),
        "notes": job.get("notes", ""),
    }


def write_snapshot(snapshot: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    for job in snapshot["jobs"]:
        slug = job["ref"].replace("/", "__")
        (out_dir / f"{slug}.log").write_text(job.get("log_tail", "") + "\n")

    history_path = out_dir / "history.jsonl"
    with history_path.open("a") as handle:
        handle.write(json.dumps({
            "checked_at": snapshot["checked_at"],
            "jobs": [
                {
                    "name": job["name"],
                    "ref": job["ref"],
                    "status": job["status"],
                }
                for job in snapshot["jobs"]
            ],
        }) + "\n")


def print_dashboard(snapshot: dict, show_logs: bool) -> None:
    print(f"\nKaggle teacher job monitor | {snapshot['checked_at']}")
    print("-" * 132)
    print(f"{'status':<11} {'progress':<32} {'fold':<9} {'epoch':<9} {'last F1':<9} {'job':<28} ref")
    print("-" * 132)
    for job in snapshot["jobs"]:
        state = job.get("progress_state") or {}
        percent = float(state.get("percent", 0.0))
        bar = f"{progress_bar(percent)} {percent:5.1f}%"
        fold = state.get("current_fold")
        completed_folds = state.get("completed_folds", [])
        fold_text = "-" if fold is None else str(fold)
        if completed_folds:
            fold_text = f"{fold_text} ({len(completed_folds)} done)"
        epoch = state.get("current_epoch", 0)
        epoch_total = state.get("current_epoch_total", "")
        epoch_text = "-" if not epoch_total else f"{epoch}/{epoch_total}"
        last_f1 = state.get("last_f1")
        f1_text = "-" if last_f1 is None else f"{last_f1:.4f}"
        print(f"{job['status']:<11} {bar:<32} {fold_text:<9} {epoch_text:<9} {f1_text:<9} {job['name'][:28]:<28} {job['ref']}")
    print("-" * 132)

    for job in snapshot["jobs"]:
        if job["progress"]:
            print(f"\n{job['name']} progress signals:")
            for line in job["progress"]:
                print(f"  {line}")
        if show_logs and job["log_tail"]:
            print(f"\n{job['name']} log tail:")
            print(job["log_tail"])


def download_completed(snapshot: dict) -> None:
    for job in snapshot["jobs"]:
        if job["status"] not in {"COMPLETE", "SUCCEEDED"}:
            continue
        output_dir = job.get("output_dir")
        if not output_dir:
            continue
        out_path = ROOT / output_dir
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {job['ref']} -> {out_path}")
        code, raw = run_kaggle(["kernels", "output", job["ref"], "-p", str(out_path)], timeout=600)
        print(raw)
        if code != 0:
            print(f"Download failed for {job['ref']} with exit code {code}", file=sys.stderr)


def run_once(args: argparse.Namespace) -> dict:
    jobs = load_jobs(args.config)
    snapshot = {
        "checked_at": utc_now(),
        "jobs": [
            check_job(job, args.log_lines, not args.no_logs)
            for job in jobs
        ],
    }
    write_snapshot(snapshot, args.out_dir)
    print_dashboard(snapshot, args.show_logs)
    if args.download_completed:
        download_completed(snapshot)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Kaggle teacher training jobs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--watch", action="store_true", help="Poll until all jobs are terminal.")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between watch checks.")
    parser.add_argument("--log-lines", type=int, default=120)
    parser.add_argument("--show-logs", action="store_true", help="Print log tails in the terminal.")
    parser.add_argument("--no-logs", action="store_true", help="Skip fetching logs; status only.")
    parser.add_argument("--download-completed", action="store_true")
    args = parser.parse_args()

    terminal = {"COMPLETE", "SUCCEEDED", "ERROR", "CANCELLED", "NOT_FOUND", "STATUS_ERROR"}
    while True:
        snapshot = run_once(args)
        statuses = {job["status"] for job in snapshot["jobs"]}
        if not args.watch or statuses.issubset(terminal):
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
