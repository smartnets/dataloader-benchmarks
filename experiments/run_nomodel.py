import subprocess
import os
from src.utils.setup_s3cmd import setup_s3cmd
from src.utils.experiments import cleanup_runs_files
from src.config import settings as st
from pathlib import Path


LIBRARIES = ["ffcv", "squirrel", "torchdata", "pytorch", "hub", "hub3", "webdataset"]


def set_general():
    os.environ["DYNACONF_DATASET"] = "random"
    os.environ["DYNACONF_BATCH_SIZE"] = "64"
    os.environ["DYNACONF_NUM_WORKERS"] = "0"
    os.environ["DYNACONF_CUTOFF"] = "-1"
    os.environ["DYNACONF_NUM_EPOCHS"] = "1"
    os.environ["DYNACONF_NAME"] = "compare_run_model"
    os.environ["DYNACONF_REMOTE"] = "False"

    st.REMOTE = False

def set_configs(num):

    


# %%
if __name__ == "__main__":

    set_general()

    for lib in LIBRARIES:


        print(f"Starting to prepare dataset of {lib}")
        ARGS = [
            "python",
            "-Wignore",
            "src/datasets/prepare.py",
            "--library",
            lib,
            "--dataset",
            "random",
        ]
        pid = subprocess.Popen(ARGS)
        pid.wait()

        print(f"Finished generating dataset: {lib}")

        for run_model in [True, False]:
            print(lib, run_model)

            cleanup_runs_files()
            path = Path(st.local_results_dir) / f"{lib}_{run_model}.txt"
            if path.is_file():
                continue

            os.environ["DYNACONF_LIBRARY"] = lib
            os.environ["IS_CUTOFF_RUN_MODEL"] = str(run_model)

            setup_s3cmd()

            # Run the experiment
            ARGS = ["python", "src/run.py"]
            try:
                print("Starting the run")
                pid2 = subprocess.Popen(ARGS)
                pid2.wait()
            except Exception as e:
                print(e)
                pid2.kill()

            print(f"Finished running dataset: {lib} {run_model}")

            path.touch()
