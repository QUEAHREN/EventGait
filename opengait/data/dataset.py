import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
import random

import numpy as np


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.rgb_in_use = data_cfg.get('rgb_in_use')  # 是否使用rgb模态
        # [label, type, view, paths]
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)     # 这个列表是用来缓存数据？
        self.indices_dict = {label: [] for label in self.label_set}
        # {'001': [0, 1, 2, 3, 4, 5, 6, 7, 8], '002': [9, 10, 11, 12, 13, 14, 15, 16], ...}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)        # 每个label的索引映射,从0开始编号
        if self.cache:      # cache会让所有数据缓存起来，这样会占用大量内存
            self.__load_all_data()
        # with open('normal_to_lowlight_mapping.json', 'r') as f:
        #     self.mapping = json.load(f)

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        if len(paths) == 0:
            raise ValueError('No data path found for loading.')
        
        for pth in paths:
            if pth.endswith('.pkl'):        # 每个pkl文件被认为是一个数据
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)      # nums 128 128 3
                f.close()
                d = _
                if self.rgb_in_use is not None:
                    if 'lowlight-voxel' in pth:
                        rgb_pth = pth.replace('/lowlight-voxel', '').replace('voxel', 'rgb')
                    else:
                        rgb_pth = pth.replace('voxel', 'rgb')
                    assert os.path.exists(rgb_pth), f'rgb path not exists: {rgb_pth}'
                    with open(rgb_pth, 'rb') as f:
                        rgb_data = pickle.load(f)      # nums 128 128 3
                    f.close()
                    if _.shape[0] != rgb_data.shape[0]*6:
                        for i in range(rgb_data.shape[0]*6 - _.shape[0]):
                            _ = np.concatenate((_, _[-1:]), axis=0)
                    assert _.shape[0] == rgb_data.shape[0]*6, f'shape not match between voxel and rgb: {_.shape}, {rgb_data.shape}'
                    d = (_, rgb_data)
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(d)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        # getitem会读取指定id的所有type所有view的所有数据？可能需要看一下self.seqs_info构建过程
        # __getitem__返回的是一个id，一个type的一个view下所有文件
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])    # self.seqs_info[idx][-1]是pkl文件路径  # len(data_list) = (voxel + rgb)
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)   # folder names on disk

        # Normalize mapping between partition labels and actual folder names.
        # Common issue: partition lists zero-padded ids like '001' while folders are '1'.
        label_set_files = set(label_list)
        padded_map = {}
        for name in label_list:
            if name.isdigit():      # 这里会把数字id全fill到三位
                padded_map[name.zfill(3)] = name

        def resolve_labels(label_names):
            resolved = []
            for lab in label_names:
                if lab in label_set_files:
                    resolved.append(lab)
                elif lab in padded_map:
                    resolved.append(padded_map[lab])
                else:
                    # try matching by integer value as a last resort
                    try:
                        lab_int = int(lab)
                        for cand in label_list:
                            if cand.isdigit() and int(cand) == lab_int:
                                resolved.append(cand)
                                break
                    except Exception:
                        pass
            return resolved

        train_set = resolve_labels(train_set)   # 规范化
        test_set = resolve_labels(test_set)

        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]      # 既不在训练集又不在测试集的id
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]      # 三个目录
                        dataset_root_use = dataset_root
                        seq_path = osp.join(dataset_root_use, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))     # 这是一个id，一个type的一个view下所有文件
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:         # mask
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]       # 这个mask顺序是按sorted的顺序来的，没有固定规则
                            seqs_info_list.append([*seq_info, seq_dirs])    # seq_dirs是一个id，一个type的一个view下所有文件
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
