import os
from sys import prefix

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2
import skvideo.io
import csv
import math


def read_float_with_comma(num):
    return float(num.replace(",", "."))



class VideoDataset_NR_LSVQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, is_test_1080p=False):
        super(VideoDataset_NR_LSVQ_SlowFast_feature, self).__init__()
        if is_test_1080p:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
        else:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20

        filename = os.path.join(self.videos_dir, video_name + '.mp4')

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_clip = int(video_length / video_frame_rate)

        video_clip_min = 8

        video_length_clip = self.num_frame

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name



class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, database_name, data_dir, filename_path, transform, resize, num_frame):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()

        if database_name == 'LIVEHFR':
            dataInfo = pd.read_csv(filename_path)
            self.video_names = dataInfo['filename']+'.mp4'
            self.score = dataInfo['MOS']
        
        elif database_name == 'ETRI':
            dataInfo = pd.read_csv(filename_path)
            self.video_names = dataInfo['new_videoname']
            self.score = dataInfo['MOS']

        elif database_name == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']

        elif database_name == 'KoNViD-1k':
            m = scio.loadmat(filename_path)
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                score.append(m['scores'][i][0])
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']

        elif database_name == 'youtube_ugc':
            m = scio.loadmat(filename_path)
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_names'][i][0][0])
                score.append(m['scores'][0][i])
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']
            
        elif database_name == 'CVD2014':
            file_names = []
            mos = []
            openfile = open("data/CVD2014_Realignment_MOS.csv", 'r', newline='')
            lines = csv.DictReader(openfile, delimiter=';')
            for line in lines:
                if len(line['File_name']) > 0:
                    file_names.append(line['File_name'])
                if len(line['Realignment MOS']) > 0:
                    mos.append(read_float_with_comma(line['Realignment MOS']))
            dataInfo = pd.DataFrame(file_names)
            dataInfo['MOS'] = mos
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'] + ".avi"
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']
        elif database_name == 'LIVEYTGaming':
            m = scio.loadmat(filename_path)
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                video_names.append(m['video_list'][i][0][0] + '.mp4')
                score.append(m['MOS'][i][0])
            dataInfo = pd.DataFrame(video_names)
            dataInfo['score'] = score
            dataInfo.columns = ['file_names', 'MOS']
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']

        elif database_name == 'waterloo':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['path']
            video_names_list = []
            for i in range(dataInfo.shape[0]):
                video_name = video_names[i].split('/')[-3]+'/'+video_names[i].split('/')[-2]+'/'+video_names[i].split('/')[-1]
                video_names_list.append(video_name)
            dataInfo['vpath'] = video_names_list
            self.video_names = dataInfo['vpath']
            self.score = dataInfo['MOS']
        
        self.database_name = database_name
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)
        self.num_frame = num_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        # video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))

        filename = os.path.join(self.videos_dir, video_name)
        print(filename)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        print(video_length)
        print(video_frame_rate)

        video_clip = int(video_length / video_frame_rate)

        # one clip for one second video
        if video_frame_rate == 0:
            video_clip_min = 10
        else:
            video_clip_min = int(video_length / video_frame_rate)

        if self.database_name == 'KoNViD-1k' or self.database_name == 'LIVEYTGaming':
            video_clip_min = 8
        elif self.database_name == 'LiveVQC' or self.database_name == 'CVD2014':
            video_clip_min = 10
        elif self.database_name == 'youtube_ugc':
            video_clip_min = 20
        elif self.database_name == 'LIVEHFR':
            video_clip_min = 6
        elif self.database_name == 'ETRI':
            video_clip_min = 5

        video_length_clip = self.num_frame

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name


class VideoDataset_LQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, database_name, data_dir, filename_path, transform, resize, num_frame):
        super(VideoDataset_LQ_SlowFast_feature, self).__init__()
        if database_name == 'LIVE-Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            self.video_names = dataInfo['file_names']
            self.score = dataInfo['MOS']
            
        elif database_name == 'BVISR':
            m_file = scio.loadmat(filename_path)
            video_names = []
            MOS = []
            for i in range(len(m_file['MOS'])):
                video_names.append(m_file['seqName'][i][0][0])
                MOS.append(m_file['MOS'][i][0])
            dataInfo = pd.DataFrame({'video_names':video_names, 'MOS':MOS})
            self.video_names = dataInfo['video_names']
            self.score = dataInfo['MOS']

        elif database_name == 'BVIHFR':
            dataInfo = pd.read_csv(filename_path)
            self.video_names = dataInfo['file_name']
            self.score = dataInfo['MOS']
        
        elif database_name == 'LIVE-livestreaming':
            dataInfo = pd.read_csv(filename_path)
            self.video_names = dataInfo['video']
            self.score = dataInfo['MOS']

        self.database_name = database_name
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)
        self.num_frame = num_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        # video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))

        filename = os.path.join(self.videos_dir, video_name)
        print(filename)

        if self.database_name == 'LIVE-Qualcomm':
            video_clip_min = 15
            video_height = 1080  # the heigh of frames
            video_width = 1920  # the width of frames
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuvj420p'})
            video_frame_rate = video_data.shape[0] // 15

        elif self.database_name == 'BVI-SR':
            video_clip_min = 5
            video_height = 2160  # the heigh of frames
            video_width = 3840  # the width of frames
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuv420p'})
            video_frame_rate = 60

        elif self.database_name == 'BVIHFR':
            video_clip_min = 10
            video_height = 1080  # the heigh of frames
            video_width = 1920  # the width of frames
            filename = os.path.join(self.videos_dir, video_name) + '-360-1920x1080.yuv'
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuv420p'})

            video_frame_rate = int(video_name.split('-')[1][:-3])

        elif self.database_name == 'LIVE-livestreaming':
            video_clip_min = 7
            video_height = 2160  # the heigh of frames
            video_width = 3840  # the width of frames
            filename = os.path.join(self.videos_dir, video_name)
            video_data = skvideo.io.vread(filename, video_height, video_width,
                                      inputdict={'-pix_fmt': 'yuv420p'})

            video_list = video_name.split('_')
            for name_str in video_list:
                if name_str[-3:]=='fps':
                    video_frame_rate = int(name_str[:-3])

        video_length = video_data.shape[0]
        video_channel = 3

        print(video_length)
        print(video_frame_rate)

        video_clip = int(video_length / video_frame_rate)

        video_length_clip = self.num_frame

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            frame = video_data[i]
            read_frame = Image.fromarray(frame)
            read_frame = self.transform(read_frame)
            transformed_frame_all[video_read_index] = read_frame
            video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]


        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name