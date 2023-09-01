import os
from datetime import datetime


def get_log_path(path=None, i=1):
    path = get_storage_path() if path is None else path
    dir_path = os.path.join(path, str(i))
    if not os.path.exists(dir_path):
        return dir_path
    else:
        if i > 30:
            return None
        return get_log_path(path, i+1)


def get_save_path(path=None, i=1):
    path = get_storage_path() if path is None else path
    dir_path = path + '_' + str(i)
    if not os.path.exists(dir_path):
        return dir_path
    else:
        if i > 30:
            return None
        return get_save_path(path, i+1)


def get_storage_path():
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d.%m")
    return os.path.join(os.path.dirname(module_path), "train", "storage", formatted_date)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
