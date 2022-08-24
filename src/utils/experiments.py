from src.config import settings as st
from pathlib import Path


def get_run_string(run_id):
    return f"{st.library:<10}, {st.num_workers:<2}, {st.batch_size:<3}, {st.remote:<5}, {st.distributed:<5}, {st.filtering:<5} {run_id}"


def record_new_run(run_id):
    string = get_run_string(run_id)
    with open(st.record_new_runs_file, "a") as fh:
        fh.write(string + "\n")


def record_completed_run(run_id):
    string = get_run_string(run_id)
    with open(st.record_completed_runs_file, "a") as fh:
        fh.write(string + "\n")


def cleanup_runs_files():
    Path(st.record_new_runs_file).unlink(missing_ok=True)
    Path(st.record_completed_runs_file).unlink(missing_ok=True)
