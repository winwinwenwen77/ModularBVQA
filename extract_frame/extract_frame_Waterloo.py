import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio


def extract_frame(videos_dir, video_name, save_folder):
    try:
        filename = os.path.join(videos_dir, video_name)
        # video_name_str = video_name
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames
        print(filename)
        print('video_length: {}, video_frame_rate: {}, video_height: {}, video_width: {}'.format(video_length, video_frame_rate, video_height, video_width))


    except:
        print(filename)

    else:
        # dim = (video_width_resize, video_height_resize)
        
        video_name_str = video_name.split('/')[-3]+'_'+video_name.split('/')[-2]+'_'+video_name.split('/')[-1]

        video_read_index = 0

        frame_idx = 0

        video_length_min = 10


        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate/2)):
                    read_frame = frame
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str,
                                             '{:03d}'.format(video_read_index) + '.png'), read_frame)
                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str,
                                         '{:03d}'.format(i) + '.png'), read_frame)

        return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return

if __name__ == '__main__':
    videos_dir = 'data/Waterloo_IVC_4K/' 
    filename_path = 'data/waterloo_data.csv'
    save_folder = 'data/waterloo_fps1'

    dataInfo = pd.read_csv(filename_path)
    video_names = dataInfo['path'].tolist()

    n_video = len(video_names)
    print(n_video)

    for i in range(n_video):
        video_name = video_names[i].split('/')[-3]+'/'+video_names[i].split('/')[-2]+'/'+video_names[i].split('/')[-1]
        print(video_name)
        extract_frame(videos_dir, video_name, save_folder)

