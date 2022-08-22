from distutils.util import strtobool
from src.config import settings as st


def should_run_test():
    return st.dataset == "cifar10"


def config_to_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return strtobool(value)
    else:
        raise ValueError("Wrong parameter")


def config_to_list(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        value = eval(value)
        assert isinstance(value, list)
        return value
    else:
        raise ValueError("Wrong paramter")
