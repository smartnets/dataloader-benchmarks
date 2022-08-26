from src.config import settings as st
from pathlib import Path
from src.utils.general import config_to_bool


def get_run_string():
    return f"{st.library:<10}, {st.num_workers:<2}, {st.batch_size:<3}, {st.rep:<2}, {st.remote:<5}, {st.distributed:<5}, {st.filtering:<5}"


def get_path(status):
    if status == "new":
        path = Path(st.record_new_runs_file)
    elif status == "completed":
        path = Path(st.record_completed_runs_file)

    if config_to_bool(st.filtering):
        mode = "filtering"
    elif config_to_bool(st.distributed):
        mode = "distributed"
    else:
        mode = "default"

    path.mkdir(exist_ok=True, parents=True)
    path /= f"{st.dataset}_{mode}.txt"
    return path


def record_run(status):
    string = get_run_string()
    path = get_path(status)
    with open(path, "a") as fh:
        fh.write(string + "\n")


def cleanup_runs_files():
    get_path("new").unlink(missing_ok=True)
    get_path("completed").unlink(missing_ok=True)


def is_run_completed():
    string = get_run_string()
    path = get_path("completed")
    try:
        with open(path, "r") as fh:
            for l in fh.readlines():
                if string == l.strip():
                    print("Simulation already completed")
                    return True
    except FileNotFoundError:
        return False
    return False
