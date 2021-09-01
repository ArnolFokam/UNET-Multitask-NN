import os


def check_or_create(path):
    """Checks if a path exists and creates it if it does not exits.
    Args:
        :param path: path to check.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path