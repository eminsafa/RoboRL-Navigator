import os
from datetime import datetime
from typing import List


def get_model_directory(path=None, i=1):
    path = get_model_storage_path() if path is None else path
    dir_path = path + '_' + str(i)
    if not os.path.exists(dir_path):
        return dir_path
    else:
        if i > 30:
            return None
        return get_model_directory(path, i + 1)


def get_model_storage_path():
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    current_date = datetime.now()
    formatted_date = current_date.strftime("%b_%d").upper()
    return os.path.join(os.path.dirname(module_path), "models", "roborl-navigator", formatted_date)


def get_assets_path(sub_directories: List[str] = []):
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(os.path.dirname(module_path), *sub_directories)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
