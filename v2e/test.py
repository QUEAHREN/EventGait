import pickle
from tqdm import tqdm
import os

path_comm = '/path/to/SUSTech1K/data/voxel'
path_mine = '/path/to/CCGR-MINI/voxel'

def check_broken_pkl_file(path):
    broken_files = []
    print(f"Checking path: {path}")
    for id in tqdm(os.listdir(path)):
        dir_path = os.path.join(path, id)
        for type in os.listdir(dir_path):
            type_path = os.path.join(dir_path, type)
            for view in os.listdir(type_path):
                view_path = os.path.join(type_path, view)
                if len(os.listdir(view_path)) == 0 or len(os.listdir(view_path)) > 1:
                    raise ValueError(f"Unexpected number of files in {view_path}")
                file = os.listdir(view_path)[0]
                if file.endswith('.pkl'):
                    file_path = os.path.join(view_path, file)
                    try:
                        with open(file_path, 'rb') as f:
                            d = pickle.load(f)
                            if d is None:
                                raise ValueError("Unpickled data is None")
                    except Exception as e:
                        print(f"Broken file detected: {file_path}, Error: {e}")
                        broken_files.append(file_path)
    return broken_files


if __name__ == "__main__":
    broken_files_comm = check_broken_pkl_file(path_comm)
    broken_files_mine = check_broken_pkl_file(path_mine)

    print("Broken files in common path:")
    for file in broken_files_comm:
        print(file)

    print("\nBroken files in mine path:")
    for file in broken_files_mine:
        print(file)