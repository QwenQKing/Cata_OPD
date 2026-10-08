pass

import hashlib
import os
import tempfile

try:
    from hdfs_io import copy, exists, makedirs  
except ImportError:
    from .hdfs_io import copy, exists, makedirs

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"

def is_non_local(path):
    return path.startswith(_HDFS_PREFIX)

def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()

def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    pass
    
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst

def copy_to_local(src: str, cache_dir=None, filelock=".file.lock", verbose=False) -> str:
    pass
    return copy_local_path_from_hdfs(src, cache_dir, filelock, verbose)

def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock=".file.lock", verbose=False) -> str:
    pass
    from filelock import FileLock

    assert src[-1] != "/", f"Make sure the last char in src is not / because it will cause error. Got {src}"

    if is_non_local(src):
        
        if cache_dir is None:
            
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        
        filelock = md5_encode(src) + ".lock"
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if not os.path.exists(local_path):
                if verbose:
                    print(f"Copy from {src} to {local_path}")
                copy(src, local_path)
        return local_path
    else:
        return src
