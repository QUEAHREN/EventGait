# process sustech1k datasets
# output: The original event stream with completed conversion (h5) and the event image with completed cropping (pkl)
import sys
import os
import json
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将文件夹B的路径添加到sys.path
v2e_dir = [os.path.join(parent_dir, 'v2e'), os.path.join(parent_dir, 'v2e/v2ecore')]
for _p in v2e_dir:
    sys.path.append(_p)
import pickle
from tqdm import tqdm
import numpy as np
import h5py
from utils import Logger, prepare_directory, Crop, Resize, create_args
import subprocess
import torch
# 文件路径
# |-SUSTech1K
#    |-images               <-- ROOT_PATH
#       |-0001
#          |-nm
#             |-90
#    |-labels
#       |-0001
#          |-nm
#             |-90
#    |-data
#       |-event             <-- OUTPUT_EVENTS_PATH
#          |-0001
#             |-nm
#                |-90
#       |-voxel             <-- OUTPUT_VOXEL_PATH
#          |-0001
#             |-nm
#                |-90

ROOT_PATH = '/path/to/CCGR-MINI/images'
OUTPUT_EVENTS_PATH = '/path/to/CCGR-MINI/lowlight/event'      # 保存v2e处理后的h5文件
OUTPUT_VOXEL_PATH = '/path/to/CCGR-MINI/lowlight/voxel'       # 保存从h5文件中生成的voxel到pkl文件

logger = Logger('sustech1k')


pad = 3


def interpolate_bboxes(original_masks, scale_factor=6):
    """
    线性插值生成扩帧后的锚框列表
    original_masks: 原始锚框列表，每个元素为(x, y, w, h)
    scale_factor: 扩帧倍数（默认6倍）
    return: 扩帧后的锚框列表
    """
    expanded_masks = []
    n_original = len(original_masks)
    
    for i in range(n_original - 1):
        curr_frame = original_masks[i]
        next_frame = original_masks[i + 1]
        
        expanded_masks.append(curr_frame)
        
        for j in range(1, scale_factor):
            alpha = j / scale_factor 
            
            # 线性插值每个分量
            x = curr_frame[0] * (1 - alpha) + next_frame[0] * alpha
            y = curr_frame[1] * (1 - alpha) + next_frame[1] * alpha
            w = curr_frame[2] * (1 - alpha) + next_frame[2] * alpha
            h = curr_frame[3] * (1 - alpha) + next_frame[3] * alpha
            
            expanded_masks.append((int(x), int(y), int(w), int(h)))
    
    # 处理最后一组帧（复制最后一帧）
    last_frame = original_masks[-1]
    expanded_masks.append(last_frame)  # 原始最后一帧
    for _ in range(1, scale_factor):
        expanded_masks.append(last_frame)  
    expanded_masks = [(x+pad, y+pad, w-2*pad, h-2*pad) for (x, y, w, h) in expanded_masks]
    return expanded_masks


def get_frame_from_h5(target_size, event_path, h5_name, label_path, scale:int=6):
    # ori_info = (W, H, fps, fps_count)
    if not os.path.exists(OUTPUT_EVENTS_PATH):
        logger.write(f"[ERROR❌] No h5 files found in {OUTPUT_EVENTS_PATH}. Please run the preprocessing step first.")
        quit(1)


    logger.print("event_path", event_path)
    H, W, fps = 128, 128, 30
    with h5py.File(os.path.join(event_path, h5_name), 'r') as f:
        events = f['events'][:].astype(np.float32)
    # load label and frame_counts == len(labels)
    mask = []
    for label_file in sorted(os.listdir(label_path)):
        with open(os.path.join(label_path, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                x, y, w, h = map(int, parts)
                mask.append((x, y, w, h))
    if len(mask) == 0:
        logger.write(f"[ERROR❌] No labels found in {label_path}, skipping...")
        quit(1)
    logger.write(f"[INFOℹ️] Loaded {len(mask)} labels from {label_path}")
    total_frames = len(mask) * scale - (scale - 1)
    mask = interpolate_bboxes(mask, scale_factor=scale)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    events_tensor = torch.from_numpy(events).to(device)
    event_frames = torch.zeros((total_frames, H, W, 3), dtype=torch.uint8, device=device)
    frame_interval = (1 / fps / scale) * 1e6  
    init_time = events[0][0]
    events[:, 0] -= init_time  # 将时间戳归一化到0开始
    events_num = len(events)
    events_counts = 0

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames)):
            start_time = frame_idx * frame_interval
            end_time = (frame_idx + 1) * frame_interval
            
            # 提取当前时间窗口内的所有事件
            time_mask = (events_tensor[:, 0] >= start_time) & (events_tensor[:, 0] < end_time)
            events_in_window = events_tensor[time_mask]
            
            if len(events_in_window) == 0:
                print(f"[WARNING⚠️] No events found for frame {frame_idx}")
            events_counts += len(events_in_window)

            coords = events_in_window[:, 1:3].long()  # x, y 坐标
            polarities = events_in_window[:, 3].byte()  # 极性 (0或1)
            valid_mask = (coords[:, 0] < W) & (coords[:, 1] < H) & (coords >= 0).all(dim=1)

            valid_coords = coords[valid_mask]
            valid_pol = polarities[valid_mask]

            pos_events = valid_pol == 1
            neg_events = ~pos_events
            
            event_frames[frame_idx, valid_coords[pos_events, 1], valid_coords[pos_events, 0], 0] += 1
            event_frames[frame_idx, valid_coords[neg_events, 1], valid_coords[neg_events, 0], 2] += 1
            
        # check
        if events_counts != events_num:
            logger.write(f"[WARNING⚠️] The number of events in the video file does not match the number of events in the mask file.")


    event_frames = torch.clip(event_frames, 0, 255).cpu().numpy()
    # mask and resize
    resized_event_frames = np.full((total_frames, target_size[1], target_size[0], 3), 0, dtype=np.uint8)
    for id in range(event_frames.shape[0]):
        resized_event_frames[id], _mask = Resize(Crop(event_frames[id], *mask[id]), target_size)
    # save voxel
    save_path = event_path.replace('event', 'voxel')
    prepare_directory(save_path, clear=False)
    pickle.dump(np.asarray(resized_event_frames), open(os.path.join(save_path, h5_name.replace('.h5', '.pkl')), 'wb'))
    logger.write(f"[INFOℹ️] Successfully extracted and saved {total_frames} frames to {save_path}")



@logger.timeit
def preprocess(target_size = (128, 128), typed='bright', scale:int=6, number:int=0):
    """
        读取图片文件夹路径启动v2e进行预处理
    """
    with open('read_list_sustech1k.json', 'r') as fp:
        read_list = json.load(fp)[str(number)]
    print(read_list, len(read_list))
    sample_list = sorted(os.listdir(ROOT_PATH))
    sample_list = [sample for sample in sample_list if sample in read_list]
    logger.write(f'[DEBUG🐞] The number of samples to be processed: {sample_list, len(sample_list)}')
    for sample in tqdm(sample_list, total=len(sample_list), desc="Processing samples"):
        sample_path = os.path.join(ROOT_PATH, sample)
        if len(os.listdir(sample_path)) == 0:
            logger.write(f"[WARNING⚠️] No pos found in {sample_path}, skipping...")
            continue
        for type in os.listdir(sample_path):
            sample_type_path = os.path.join(sample_path, type)
            for angle in os.listdir(sample_type_path):
                images_sample_pos_angle_path = os.path.join(sample_type_path, angle)
                if len(os.listdir(images_sample_pos_angle_path)) <= 2:
                    logger.write(f"[WARNING⚠️] No images found or too few images in {images_sample_pos_angle_path}, skipping...")
                    continue
                labels_sample_pos_angle_path = images_sample_pos_angle_path.replace('images', 'labels')
                event_path = os.path.join(OUTPUT_EVENTS_PATH, sample, type, angle)
                prepare_directory(event_path, clear=True)
                args_namespace, other_args, command_line = create_args(
                                    input_path = images_sample_pos_angle_path, 
                                    output_path = event_path,
                                    output_h5 = f'{sample}_{type}_{angle}.h5',
                                    input_frame_rate = 30,
                                    output_width = 128,
                                    output_height = 128,
                                    batch_size = 64,
                                    target_rate=180. ,
                                    type=typed
                                )
                logger.write(f"[INFOℹ️] Processing {sample}_{type}_{angle}...")
                try:
                    v2e_command = ['v2e.py']
                    v2e_command.extend(args_namespace)  
                    print(v2e_command)
                    subprocess.run(v2e_command)
                #    v2e_main(args_namespace, other_args, command_line)
                except Exception as e:
                    logger.write(f"[ERROR❌] Failed to process {sample}_{type}_{angle}: {e}")
                    quit()
                torch.cuda.empty_cache()
                logger.write(f"[INFOℹ️] Successfully processed {sample}_{type}_{angle}, now extracting frames...")
                get_frame_from_h5(target_size, event_path, f'{sample}_{type}_{angle}.h5', labels_sample_pos_angle_path, scale=scale)





if __name__ == "__main__":
    preprocess(typed='dark', number=3)