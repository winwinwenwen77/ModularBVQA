import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio

def extract_frame(videos_dir, video_name, save_folder):
    try:
        filename = os.path.join(videos_dir, video_name)
        video_name_str = video_name[:-4]
        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))


        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames


    except:
        print('fails:', filename)

    else:
        # dim = (video_width_resize, video_height_resize)

        video_read_index = 0

        frame_idx = 0

        video_length_min = 10

        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                # if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate / 2)):
                if (video_read_index < video_length) and (
                        frame_idx % (int(2*video_frame_rate)) == int(video_frame_rate)):
                    # read_frame = cv2.resize
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
    videos_dir = 'data/YouTube-UGC/original_videos_h264'
    filename_path = 'data/youtube_ugc_data.mat'
    save_folder = 'data/youtube_ugc_image_all_fps05'

    dataInfo = scio.loadmat(filename_path)
    n_video = len(dataInfo['video_names'])
    video_names = []

    for i in range(n_video):
        video_names.append(dataInfo['video_names'][i][0][0])


    for i in range(n_video):
        print(i)
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame(videos_dir, video_name, save_folder)
