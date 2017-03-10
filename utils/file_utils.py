import os
import errno


def ensure_dir_exists(path, dir_type=True, mode=0o755):
    """ensure that a directory exists

    If it doesn't exist, try to create it and protect against a race condition
    if another process is doing the same.

    Args:
        path: str, the path to check
        dir_type: If 'dir_type' is true, the path is a directory, else the path is file
        mode: The default permissions are 755, which differ from os.makedirs default of 777.
    """
    path = path if dir_type else os.path.dirname(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise IOError("%r exists but is not a directory" % path)