# process CCGR-MINI datasets
# output: The original event stream with completed conversion (h5) and the event image (h5)
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将文件夹B的路径添加到sys.path
v2e_dir = [os.path.join(parent_dir, 'v2e'), os.path.join(parent_dir, 'v2e/v2ecore')]
for _p in v2e_dir:
    sys.path.append(_p)
import cv2
from tqdm import tqdm
import numpy as np
import h5py
from utils import Logger, prepare_directory, Crop, Resize, create_args, get_video_info, mask_function
import pickle
import argparse
import json
import subprocess
import torch

ROOT_PATH = '/path/to/SUSTech1K2'
OUTPUT_EVENTS_PATH = '/path/to/DVS128/images/event'      # 保存v2e处理后的h5文件
OUTPUT_VOXEL_PATH = '/path/to/DVS128/images/voxel'      # 保存从h5文件中生成的voxel到pkl文件

logger = Logger('ccgr')


pad = 3


    



def get_ccgr_mask(path, threshold=10):
    """获取非黑色填充区域的边界框坐标
    Args:
        path: 视频的路径
        threshold: 黑色像素的阈值(0-255)
    Returns:
        (x, y, w, h) 非黑色区域的边界框坐标
        如果全是黑色内容则返回None
    """
    video = cv2.VideoCapture(path)
    mask_list = []
    # video_with_mask = cv2.VideoWriter('./mask_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (286, 252))
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        mask = mask_function(frame, threshold)
        if mask is None:
            logger.write(f"[ERROR❌] No mask found for video {path}")
            continue
        mask_list.append(mask)
        # x,y,w,h = mask
        # 将mask可视化到视频
        # print(x, y, w, h)
        # mask_frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # video_with_mask.write(mask_frame)
    
    video.release()
    # video_with_mask.release()
    return mask_list


def interpolate_bboxes(original_masks, scale_factor=6):
    """
    插值生成扩帧后的锚框列表
    :param original_masks: 原始锚框列表，每个元素为(x, y, w, h)
    :param scale_factor: 扩帧倍数（默认6倍）
    :return: 扩帧后的锚框列表
    """
    expanded_masks = []
    n_original = len(original_masks)
    
    for i in range(n_original - 1):
        # 当前原始帧（关键帧）
        curr_frame = original_masks[i]
        # 下一原始帧
        next_frame = original_masks[i + 1]
        
        # 添加关键帧（位置0）
        expanded_masks.append(curr_frame)
        
        # 在当前帧和下一帧之间插值
        for j in range(1, scale_factor):
            alpha = j / scale_factor  # 插值比例
            
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
        expanded_masks.append(last_frame)  # 复制5次
    expanded_masks = [(x+pad, y+pad, w-2*pad, h-2*pad) for (x, y, w, h) in expanded_masks]
    
    return expanded_masks

def get_frame_from_h5(target_size, event_path, h5_name, ori_info, mask, scale:int=6):
    # ori_info = (W, H, fps, fps_count)
    if not os.path.exists(OUTPUT_EVENTS_PATH):
        logger.write(f"[ERROR❌] No h5 files found in {OUTPUT_EVENTS_PATH}. Please run the preprocessing step first.")
        quit(1)

    W, H, fps, fps_count = ori_info
    logger.write(f"[INFOℹ️] Loading events from {event_path}")
    with h5py.File(os.path.join(event_path, h5_name), 'r') as f:
        events = f['events'][:].astype(np.float32)
    total_frames = len(mask) * scale - (scale - 1)
    # mask = [i for i in mask for _ in range(scale)]
    mask = interpolate_bboxes(mask, scale)
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
    # 保存
    save_path = event_path.replace("event", "voxel")
    prepare_directory(save_path, clear=True)
    # build per-frame names (repeat mask name for each of the scale frames)
    pickle.dump(np.asarray(resized_event_frames), open(os.path.join(save_path, os.path.splitext(h5_name)[0]+"-aligned-rgbs.pkl"), "wb"))

    logger.write(f"[INFOℹ️] Saved {total_frames} frames to {save_path}")

@logger.timeit
def preprocess(target_size = (128, 128), typed='bright', scale:int=6, number:int=0):
    """
        将图片文件夹用v2e处理成事件流，并保存成h5文件
    """
    read_list = [str(i) for i in range(850, 901)]
    sample_list = sorted(os.listdir(ROOT_PATH))
    sample_list = [sample for sample in sample_list if sample in read_list]
    print(sample_list)
    for sample in tqdm(sample_list, total=len(sample_list), desc="Processing samples"):
        sample_path = os.path.join(ROOT_PATH, sample)
        if len(os.listdir(sample_path)) == 0:
            logger.write(f"[WARNING⚠️] No pos found in {sample_path}, skipping...")
            continue

        for type in os.listdir(sample_path):
            sample_type_path = os.path.join(sample_path, type)
            for angle in os.listdir(sample_type_path):
                view = angle.split('.')[0]

                png_path = os.path.join(sample_type_path, angle)
                if len(os.listdir(png_path)) ==0:
                    logger.write(f"[ERROR❌] No png files found in {png_path}. Please check the filepath.")
                    quit(1)
                # get mask
                masks = []
                png_files = sorted([f for f in os.listdir(png_path) if f.endswith('.png')])
                for f in png_files:
                    img = cv2.imread(os.path.join(png_path, f))
                    mask = mask_function(img, threshold=10)
                    if mask is None:
                        logger.write(f"[ERROR❌] No mask found for video {os.path.join(png_path, f)}")
                        quit(1)
                    masks.append(mask)  
                filename = f"{sample}_{type}_{view}-aligned-rgbs.pkl"
                f = os.path.join(OUTPUT_VOXEL_PATH, sample, type, view, filename)

                if os.path.exists(f):
                    logger.write(f"[INFOℹ️] The file {f} is exists.continue.")
                    continue
                
                info = (128, 128, 5, len(masks))     # w, h, fps, fps_count

                event_path = os.path.join(OUTPUT_EVENTS_PATH, sample, type, view)
                prepare_directory(event_path, clear=False)
                args_namespace, other_args, command_line = create_args(
                    input_path = png_path, 
                    output_path = event_path,
                    output_h5 = f'{sample}_{type}_{view}.h5',
                    input_frame_rate = info[2],
                    output_width = 128,
                    output_height = 128,
                    batch_size = 64,
                    target_rate=info[2]*scale,
                    type=typed
                )
                logger.write(f"[INFOℹ️] Processing {sample}_{type}_{view}...")
                try:
                    v2e_command = ['v2e.py']
                    v2e_command.extend(args_namespace)  
                    print(v2e_command)
                    subprocess.run(v2e_command)
                #    v2e_main(args_namespace, other_args, command_line)
                except Exception as e:
                    logger.write(f"[ERROR❌] Failed to process {sample}_{type}_{view}: {e}")
                    quit()
                torch.cuda.empty_cache()
                logger.write(f"[INFOℹ️] Successfully processed {sample}_{type}_{view}, now extracting frames...")
                get_frame_from_h5(target_size, event_path, f'{sample}_{type}_{view}.h5', info, masks, scale=scale)



if __name__ == "__main__":
    preprocess(typed='dark', number=0)
    # get_ccgr_mask('/path/to/CCGR-MINI/CCGR-MINI-RGB-V1/991/PK1/112_5_1.avi/112_5_1rgb_f-991-PK1.avi')
    # video2frame('/path/to/CCGR-MINI/raw/1/AS1/180_1.avi/180_1rgb_f-1-AS1.avi', './test_frames')