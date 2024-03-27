import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio
import skvideo.io
from PIL import Image


def extract_frame(videos_dir, video_name, save_folder):
    try:
        video_height = 2160  # the heigh of frames
        video_width = 3840  # the width of frames

        filename = os.path.join(videos_dir, video_name)

        video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuvj420p'})

        video_name_str = video_name[:-4]

        video_length = video_data.shape[0]

        video_list = video_name_str.split('_')
        for name_str in video_list:
            if name_str[-3:]=='fps':
                video_frame_rate = int(name_str[:-3])


        print(filename)
        print(video_length)
        print(video_frame_rate)

    except:
        print(filename)

    else:

        video_read_index = 0

        frame_idx = 0

        video_length_min = 7

        for i in range(video_length):
            frame = video_data[i]
            if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate/2)):
                read_frame = frame
                read_frame = Image.fromarray(read_frame)
                exit_folder(os.path.join(save_folder, video_name_str))
                read_frame.save(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(video_read_index) + '.png'))
                video_read_index += 1
            frame_idx += 1


        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                read_frame.save(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'))

        return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return

if __name__ == '__main__':
    videos_dir = 'data/LIVE_livestreaming/yuv/'
    filename_path = 'data/LIVE_livestreaming_scores.csv'
    save_folder = 'data/livestreaming_image_all_fps1'

    dataInfo = pd.read_csv(filename_path)
    video_names = dataInfo['video'].tolist()

    n_video = len(video_names)


    for i in range(n_video):
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame(videos_dir, video_name, save_folder)
