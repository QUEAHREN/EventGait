import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all', 'evfixed', 'evall']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered', 'allordered']:
            raise ValueError
        # BUG
        # self.ordered = sample_type[1] == 'ordered'
        self.ordered = sample_type[1] == 'ordered'
        self.allordered = self.ordered and "all" in sample_type[1]      # 包括evall_ordered 和 all_ordered

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # evfixed cases
        if self.sampler == 'evfixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']
            self.frames_skip_num = sample_config['frames_skip_num']
            self.chunk_size = sample_config['chunk_size']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler not in ['all', 'evall'] and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']
        
        # evall cases
        if self.sampler == 'evall' and 'frames_all_limit' in sample_config and 'chunk_size' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']
            self.chunk_size = sample_config['chunk_size']

        self.points_in_use = sample_config.get('points_in_use')
        self.events_in_use = sample_config.get('events_in_use')
        self.rgb_in_use = sample_config.get('rgb_in_use')  # 是否使用rgb模态 


    def _pad_sequences(self, sequences, chunk_size):
        # Pad sequences to the target length with last frame
        # sequences.shape = (all_nums in pkl, 128, 128, 3)
        n = len(sequences)
        remainder = n % chunk_size
        if remainder == 0:
            return sequences
        padding_count = chunk_size - remainder
        last_frame = sequences[-1]
        padding = [last_frame] * padding_count
        return np.concatenate((sequences, padding), axis=0)



    def __call__(self, batch):      
        # batch = features, labels, seq_type, view, paths
        # print(batch)
        batch_size = len(batch)     # 128 = 8*16
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0])        
            labs_batch.append(self.label_set.index(bt[1][0]))   
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])
        global count
        count = 0
        # seqs_batch = (B, 1, all_nums in pkl, 128, 128, 3)
        def sample_frames(seqs):
            global count    # seqs.shape = (1, all_nums in pkl, 128, 128, 3)
            if self.rgb_in_use is not None:
                seq_seqs = [seqs[0][0]]     # (1, 150, 128, 128, 3)
                rgb_seqs = [seqs[0][1]]     # (1, 900, 128, 128, 3)
            else:
                seq_seqs = seqs
                rgb_seqs = None
            sampled_fras = [[] for i in range(feature_num)]
            if self.events_in_use is not None and 'ev' in self.sampler:
                events_fras = [[] for i in range(feature_num)]
            seq_len = len(seq_seqs[0])  # 一个pkl内 序列有多少个帧
            # print(np.asarray(seqs).shape)   # 1 num_frames 128 128 3
            indices = list(range(seq_len))
            # padding
            # print("Before pad:", np.asarray(seq_seqs).shape)
            for i in range(feature_num):
                if self.events_in_use is not None and 'ev' in self.sampler:
                    seq_seqs[i] = self._pad_sequences(seq_seqs[i], self.chunk_size) # _pad_sequences保证了seq_seqs[i]的长度是chunk_size的整数倍
            # print("After pad:", np.asarray(seq_seqs).shape)
            if self.sampler in ['fixed', 'unfixed', 'evfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                elif self.sampler == 'unfixed':
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))
                elif self.sampler == 'evfixed':
                    frames_num = self.frames_num_fixed
                    # 这里的indices是等间隔采样，间隔为self.chunk_size
                    indices = np.arange(0, seq_len, self.chunk_size).tolist()   # 二级索引
                    seq_len = len(indices)  # 更新seq_len为chunk采样后的长度
                    if len(indices) == 0:   # 如果chunk_size太大或pkl内帧数太少，导致indices为空。后面压帧可能还要判断是否越界
                        indices = [0]
                if self.allordered:     # only fixed_allordered or evfixed_allordered
                    replace = seq_len < frames_num  # 如果seq len比frames_num小，则随机抽样允许选择相同元素

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))
                    count += 1
                    indices = sorted(np.random.choice(
                        indices, frames_num, replace=replace))
                elif self.ordered:  # evfixed_ordered or fixed_ordered
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it      # repeat

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:       # unordered
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))
                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)
                    
            # 跨域时这段需要注释，因为跨域数据集不是高帧率采集的，直接二次压帧会导致信息丢失过多
            frames_all_limit = self.frames_all_limit
            if self.sampler in ['evall']:   # only evall
                indices = np.arange(0, seq_len, self.chunk_size).tolist()   # 一级索引
                seq_len = len(indices)  # 更新seq_len为chunk采样后的长度
                if len(indices) == 0:   # 如果chunk_size太大或pkl内帧数太少，导致indices为空。后面压帧可能还要判断是否越界
                    indices = [0]
                frames_all_limit = max(1, self.frames_all_limit // self.chunk_size)     # evall的frames_all_limit是针对二次压帧后的帧数限制

            for i in range(feature_num):
                for j in (indices[:frames_all_limit] if frames_all_limit > -1 and len(indices) > frames_all_limit else indices):
                    point_cloud_index = self.points_in_use.get('pointcloud_index') if self.points_in_use else None
                    if self.points_in_use is not None and point_cloud_index is not None and i == point_cloud_index:
                        points_num = self.points_in_use.get('points_num')
                        sample_points = (random.choices(range(len(seq_seqs[i][j])), k=points_num)
                                if points_num is not None else list(range(len(seq_seqs[i][j]))))
                        sampled_fras[i].append(np.asarray([seq_seqs[i][j][p] for p in sample_points]))
                    elif 'ev' in self.sampler:
                        # 二次压帧
                        add_frames = np.zeros_like(seq_seqs[i][0]).astype(np.float32)
                        stride = self.chunk_size
                        used_event_frames = []
                        for k in range(j, min(j + stride, len(seq_seqs[i])), 1):
                            np.add(add_frames, seq_seqs[i][k], out=add_frames)
                            used_event_frames.append(seq_seqs[i][k])
                        np.clip(add_frames, 0, 255, out=add_frames)     
                        if self.events_in_use is not None:
                            assert len(used_event_frames) == stride, \
                            "The length of used_event_frames should be equal to stride.but got {} and {}, j={}".format(
                                len(used_event_frames), stride, j)
                            events_fras[i].append(np.asarray(used_event_frames))
                        if self.rgb_in_use is not None:
                            sampled_fras[i].append((add_frames, rgb_seqs[i][j//stride]))
                        else:
                            sampled_fras[i].append(add_frames)    # every sampler = 二次压帧
                    else:
                        sampled_fras[i].append(seq_seqs[i][j])
            if self.events_in_use is not None and 'ev' in self.sampler:
                return sampled_fras, events_fras
            return sampled_fras

        # f: feature_num
        # b: batch_size = p * k
        # p: batch_size_per_gpu
        # g: gpus_num
        # sample_frames may return either sampled_fras or (sampled_fras, events_fras)
        sampled_results = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f] or [(f, f_events), ...]
        # batch now has an extra slot at the end for events_frames when events_in_use is set
        batch = [None, labs_batch, typs_batch, vies_batch, None, None, None]
        # If events_in_use and an event-type sampler, split sampled_results into fras_batch and events_batch
        rgbs_batch, events_batch = None, None
        if self.events_in_use is not None and 'ev' in self.sampler:
            fras_batch = []
            events_batch = []
            for item in sampled_results:
                # item should be a tuple (sampled_fras, events_fras) when events_in_use is enabled
                if self.rgb_in_use is not None:
                    if rgbs_batch is None:
                        rgbs_batch = []
                    fr_rgb, ev = item
                    fr, rgb = np.asarray(fr_rgb)[:,:,0,:,:,:], np.asarray(fr_rgb)[:,:,1,:,:,:]
                    fras_batch.append(fr)
                    rgbs_batch.append(rgb)
                    events_batch.append(ev)
                else:
                    if isinstance(item, tuple) or (isinstance(item, list) and len(item) == 2):
                        fr, ev = item
                    else:
                        # fallback: no events returned
                        fr, ev = item, None
                    fras_batch.append(fr)
                    events_batch.append(ev)
        else:
            if self.rgb_in_use is not None:
                fras_batch = []
                rgbs_batch = []
                for item in sampled_results:
                    fr_rgb = item
                    fr, rgb = np.asarray(fr_rgb)[:,:,0,:,:,:], np.asarray(fr_rgb)[:,:,1,:,:,:]
                    fras_batch.append(fr)
                    rgbs_batch.append(rgb)
            else:
                # normal case: each element is just sampled_fras
                fras_batch = sampled_results


        events_frames_batch, rgbs_fras_batch = None, None
        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [feature_num, batch_size]
        elif self.sampler in ["evfixed", "evall"]:
            # evfixed: if events_batch exists, also build events_frames structure
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [feature_num, batch_size]
            if events_batch is not None:
                # events_batch: [batch_size][feature_num] -> convert to [feature_num, batch_size]
                events_frames_batch = [[np.asarray(events_batch[i][j]) if events_batch[i] is not None else None
                                         for i in range(batch_size)]
                                        for j in range(feature_num)]
            if rgbs_batch is not None:
                rgbs_fras_batch = [[np.asarray(rgbs_batch[i][j]) if rgbs_batch[i] is not None else None for i in range(batch_size)]
                            for j in range(feature_num)]  # [feature_num, batch_size]
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, l]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            # seqL goes into slot index 4
            batch[4] = np.asarray(seqL_batch)      # seqL代表序列长度，只有不是fixed模式才不是None




        batch[0] = fras_batch       # (1, 1, 768, 128, 128, 4)
        batch = [batch[0], batch[1], batch[2], batch[3], batch[4]]
        if events_frames_batch is not None:
            batch.append(events_frames_batch)
        if rgbs_fras_batch is not None:
            batch.append(rgbs_fras_batch)
        return batch
