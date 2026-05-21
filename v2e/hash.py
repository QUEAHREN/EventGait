import os
from tqdm import tqdm

BASE = 1425

PATH_CCGR_MINI_DARK = '/path/to/DVS128/CCGR'

TYPE = ['event', 'voxel']

for _type in TYPE:
    dir_path = os.path.join(PATH_CCGR_MINI_DARK, _type)
    dir_list = sorted(os.listdir(dir_path), key = lambda x: int(x))
    for idx, dir_name in enumerate(tqdm(dir_list)):
        src = os.path.join(dir_path, dir_name)
        dst = os.path.join(dir_path, str(BASE + idx))
        os.rename(src, dst)
