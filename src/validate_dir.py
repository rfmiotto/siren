import os
import shutil
from pathlib import Path


def _empty_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as err:
            print(f"Failed to delete {file_path}. Reason: {err}")


def validate_dir(path: str, create: bool = True):
    path = Path(path).resolve()
    if path.exists():
        # val = input(f"The directory {log_dir} exists. Overwrite? ([y]/n)")
        # if val in ["", "y", "Y"]:
        _empty_folder(path)
        return
    if not path.exists() and create:
        path.mkdir(parents=True)
    else:
        raise NotADirectoryError(f"log_dir {path} does not exist.")
