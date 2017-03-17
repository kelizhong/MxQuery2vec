from utils.file_util import ensure_dir_exists
import logging
import pickle
import os


def save_obj_pickle(obj, filename, overwrite=False):
    ensure_dir_exists(filename, is_dir=False)
    if os.path.isfile(filename) and not overwrite:
        logging.warning("Not saving %s, already exists." % filename)
    else:
        if os.path.isfile(filename):
            logging.info("Overwriting %s." % filename)
        else:
            logging.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
