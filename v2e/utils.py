import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将文件夹B的路径添加到sys.path
v2e_dir = [os.path.join(parent_dir, 'v2e'), os.path.join(parent_dir, 'v2e/v2ecore')]
for _p in v2e_dir:
    sys.path.append(_p)
import cv2
import logging
import argparse
from v2e_args import v2e_args
import time
import torch
import h5py
import numpy as np
import random

def prepare_directory(path, clear=False):
    if clear and os.path.exists(path):
        os.system(f"rm -rf {path}")  
    os.makedirs(path, exist_ok=True)

def Resize(img, target_size=(128,128)):    
    target_h, target_w = target_size

    # 获取图像尺寸
    if len(img.shape) == 3:
        h, w, c = img.shape

    elif len(img.shape) == 2:
        h, w = img.shape

    else:
        raise ValueError("不支持的图像维度")
    scale = min(target_h / h, target_w / w)
    
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    resized_image = cv2.resize(img, (new_width, new_height), 
                               interpolation=cv2.INTER_LINEAR)
    
    top = (target_h - new_height) // 2
    bottom = target_h - new_height - top
    left = (target_w - new_width) // 2
    right = target_w - new_width - left
    
    framed_img = cv2.copyMakeBorder(resized_image, 
                                   top=top, bottom=bottom, 
                                   left=left, right=right, 
                                   borderType=cv2.BORDER_CONSTANT, 
                                   value=[0, 0, 0])
    
    return framed_img, (left, top, new_width, new_height)

def video2frame(video_path, output_folder):
    """
    将视频文件转换为帧并保存到指定文件夹

    参数:
        video_path (str): 视频文件路径
        output_folder (str): 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        frame_resized = cv2.resize(frame, (128, 128))  # Resize to 128x128
        frame_file = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_file, frame_resized)
        frame_count += 1
    video_capture.release()



def Crop(img, x, y, w, h):
    if len(img.shape) == 3:     # HWC
        cropped_image = img[int(y):int(y + h), int(x):int(x + w), :]
    elif len(img.shape) == 2:
        cropped_image = img[int(y):int(y + h), int(x):int(x + w)]
    else:
        raise ValueError("不支持的图像维度")
    return cropped_image

def create_args(input_path, output_path, output_h5, input_frame_rate, output_width, output_height, batch_size=64, target_rate=30., type='bright'):
    # dt * (2*pi*cutoff_hz) < 0.3
    # set configs
    if type == "ideal":
        thre_low, thre_high = 0.05, 0.5
        sigma_low, sigma_high = 0, 0
        cutoff_hz_low, cutoff_hz_high = 0, 0
        leak_rate_hz_low, leak_rate_hz_high = 0, 0
        shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
    elif type == "bright":
        thre_low, thre_high = 0.05, 0.5
        sigma_low, sigma_high = 0.03, 0.05
        cutoff_hz_low, cutoff_hz_high = 200, 200
        leak_rate_hz_low, leak_rate_hz_high = 0.1, 0.5
        shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
    elif type == "dark":
        thre_low, thre_high = 0.05, 0.5
        sigma_low, sigma_high = 0.03, 0.05
        cutoff_hz_low, cutoff_hz_high = 10, 100
        leak_rate_hz_low, leak_rate_hz_high = 0, 0
        shot_noise_rate_hz_low, shot_noise_rate_hz_high = 1, 10

    thres = random.uniform(thre_low, thre_high)
    sigma = random.uniform(
        min(thres*0.15, sigma_low), min(thres*0.25, sigma_high)) \
        if type != "ideal" else 0
    leak_rate_hz = random.uniform(leak_rate_hz_low, leak_rate_hz_high)
    shot_noise_rate_hz = random.uniform(
        shot_noise_rate_hz_low, shot_noise_rate_hz_high)
    cutoff_hz = random.uniform(cutoff_hz_low, cutoff_hz_high) if type != 'dark' else shot_noise_rate_hz*10

    parser = argparse.ArgumentParser()
    parser = v2e_args(parser)
    # 构造参数列表
    args_list = [
        '--input', input_path,
        '--output_folder', output_path,
        '--overwrite',
        '--timestamp_resolution', str(1 / target_rate),       # mode
        '--auto_timestamp_resolution', 'false',
        '--skip_video_output',
        '--dvs_h5', output_h5,
        '--no_preview',
        '--input_frame_rate', str(input_frame_rate),    
        '--batch_size', str(batch_size),
        '--shot_noise_rate_hz', str(shot_noise_rate_hz),            # mode
        '--leak_rate_hz', str(leak_rate_hz),                   # mode
        '--cutoff_hz', str(cutoff_hz),                           # mode
        '--sigma_thres', str(sigma),                     # mode
        '--output_width', str(output_width),
        '--output_height', str(output_height),
        # "--dvs_exposure", "duration", "0.005",
    ]
    
    # args = parser.parse_args(args_list)
    other_args = []
    command_line = ' '.join(args_list)
    print(args_list)
    return (args_list, other_args, command_line)

def mask_function(image, threshold):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 二值化处理 (所有非纯黑区域变为白色)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 找到非黑色区域的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 获取最大轮廓的边界矩形
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    return (x, y, w, h)

def get_video_info(video_path):
    """
    获取视频文件的尺寸（宽度和高度）
    
    参数:
        video_path (str): 视频文件路径
        
    返回:
        tuple: (width, height, fps, frame_count) 视频的宽度和高度, fps, 总帧数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # 直接从视频属性获取尺寸（无需解码帧）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    cap.release()
    return (width, height, fps, frame_count)

class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        # if os.path.exists('../output/v2e.log'):
        #     os.remove('../output/v2e.log')

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(lineno)d %(levelname)s %(message)s'
        )

    def write(self, message):
        self.logger.info(message)

    def print(self, name, message):
        self.logger.info(f"[DEBUG🐛] {name} = {message}")

    # 写一个logger计时装饰器
    def timeit(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            self.logger.info(f"[TIME⏰] {func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper

def h52video():
    h5_path = "/path/to/CCGR-MINI/voxel/1/BG1/45_2.avi/45_2rgb_f-1-BG1.h5"
    output_video_path = 'output/event_video.avi'

    with h5py.File(h5_path, 'r') as f:
        filename = f.attrs.get('filename')
        print(filename)
        images_list = f['frames']
        frame = images_list[0]
        height, width = frame.shape
        video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
        for image in images_list:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            video.write(image)
        video.release()
        img = images_list[10]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape)
        cv2.imwrite('./img.png', img)

def frame2video():
    frame_path = "../output/"
    output_video_path = 'output/event_video.avi'
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (128, 128))
    print(sorted(os.listdir(frame_path),  key= lambda x: int(x.replace('.png', ''))))
    for img in sorted(os.listdir(frame_path),  key= lambda x: int(x.replace('.png', ''))):
        filepath = os.path.join(frame_path, img)
        img = cv2.imread(filepath)
        
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        video.write(img)
    video.release()


def pklvideo():
    import pickle
    pkl_path = "/path/to/CCGR-MINI/voxel/1/AS1/180/180_1rgb_f-1-AS1-aligned-rgbs.pkl"
    output_video_path = 'output/event_video.avi'
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
    data = pickle.load(open(pkl_path, 'rb'))
    for img in data:
        img *= 255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()


if __name__ == '__main__':
    pklvideo()