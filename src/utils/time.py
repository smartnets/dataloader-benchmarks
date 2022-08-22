import datetime


def get_current_timestamp():
    return int(datetime.datetime.utcnow().timestamp() * 1e6)
