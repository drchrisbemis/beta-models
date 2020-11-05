

import inspect
import glob
import os
import sys

import datetime as dt


def create_unique_filename(filename, extension, path):
    if not os.path.exists(path):
        os.mkdir(path)

    unique_filename = '{0}_{1}.{2}'.format(filename,
        dt.datetime.now().strftime("%Y%m%d%H%M%S"),
        extension)

    unique_filename = os.path.join(path, unique_filename)

    return unique_filename

def get_recent_file(mask):
    return max(glob.iglob(mask), key=os.path.getctime)

def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False):
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)
