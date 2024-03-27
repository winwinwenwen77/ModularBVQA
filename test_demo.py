import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model import modular
from pytorchvideo.models.hub import slowfast_r50
import cv2
from PIL import Image
from thop import profile
from torchvision import transforms, models
import time
import pandas as pd
from scipy.optimize import curve_fit


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic



def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def pyramidsGL(image, num_levels, dim=224):
    ''' Creates Gaussian (G) and Laplacian (L) pyramids of level "num_levels" from image im. 
    G and L are list where G[i], L[i] stores the i-th level of Gaussian and Laplacian pyramid, respectively. '''
    o_width = image.shape[1]
    o_height = image.shape[0]
    if o_width>(dim+num_levels) and o_height>(dim+num_levels) :
        if o_width > o_height:
            f_height = dim
            f_width = int((o_width*f_height)/o_height)
        elif o_height > o_width:
            f_width = dim
            f_height = int((o_height*f_width)/o_width)
        else:
            f_width = f_height = dim

        height_step = int((o_height-f_height)/(num_levels-1))*(-1)
        width_step = int((o_width-f_width)/(num_levels-1))*(-1)
        height_list = [i for i in range(o_height, f_height-1, height_step)]
        width_list = [i for i in range(o_width, f_width-1, width_step)]
    
    elif o_width==dim or o_height==dim :
        height_list = [o_height for i in range(num_levels)]
        width_list = [o_width for i in range(num_levels)]

    else:
        if o_width > o_height:
            f_height = dim
            f_width = int((o_width*f_height)/o_height)
        elif o_height > o_width:
            f_width = dim
            f_height = int((o_height*f_width)/o_width)
        else:
            f_width = f_height = dim
        image = cv2.resize(image, (f_width, f_height), interpolation = cv2.INTER_CUBIC)
        height_list = [f_height for i in range(num_levels)]
        width_list = [f_width for i in range(num_levels)]

    layer = image.copy()
    gaussian_pyramid = [layer]    #Gaussian Pyramid
    laplacian_pyramid = []         # Laplacian Pyramid

    for i in range(num_levels-1):
        blur = cv2.GaussianBlur(gaussian_pyramid[i], (5,5), 5)
        layer = cv2.resize(blur, (width_list[i+1], height_list[i+1]), interpolation = cv2.INTER_CUBIC)
        gaussian_pyramid.append(layer)

        uplayer = cv2.resize(blur, (width_list[i], height_list[i]), interpolation = cv2.INTER_CUBIC)
        laplacian = cv2.subtract(gaussian_pyramid[i], uplayer)
        laplacian_pyramid.append(laplacian)
    gaussian_pyramid.pop(-1)
    return gaussian_pyramid, laplacian_pyramid


def resizedpyramids(gaussian_pyramid, laplacian_pyramid, num_levels, width, height):
    gaussian_pyramid_resized, laplacian_pyramid_resized=[],[]
    for i in range(num_levels-1):
        img_gaussian_pyramid = cv2.resize(gaussian_pyramid[i],(width, height), interpolation = cv2.INTER_CUBIC)
        img_laplacian_pyramid = cv2.resize(laplacian_pyramid[i],(width, height), interpolation = cv2.INTER_CUBIC)
        gaussian_pyramid_resized.append(img_gaussian_pyramid)
        laplacian_pyramid_resized.append(img_laplacian_pyramid)
    return gaussian_pyramid_resized, laplacian_pyramid_resized


class ResNet18_LP(torch.nn.Module):
    """Modified ResNet18 for feature extraction"""
    def __init__(self,):
        super(ResNet18_LP, self).__init__()
        self.features = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-4])
        for p in self.features.parameters():
            p.requires_grad = False 

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
            features_std = global_std_pool2d(x)
        return torch.cat((features_mean, features_std), 1).squeeze()



def video_processing_spatial(dist):
    video_name = dist
    video_name_dis = video_name

    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    cap = cv2.VideoCapture(video_name)

    video_channel = 3

    video_height_crop = 224
    video_width_crop = 224

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 

    print('video_length:', video_length)
    print('video_frame_rate:', video_frame_rate)

    video_length_read = int(video_length / video_frame_rate)

    transformations = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), \
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

    transformations_lp = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed_lp = torch.zeros([video_length_read*(5), video_channel, video_height, video_width])

    video_read_index = 0
    frame_idx = 0

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:

            # key frame
            if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == int(video_frame_rate / 2)):
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                read_frame = transformations(read_frame)
                transformed_video[video_read_index] = read_frame

                gaussian_pyramid,laplacian_pyramid = pyramidsGL(frame, 6)
                _, laplacian_pyramid_resized = resizedpyramids(gaussian_pyramid, laplacian_pyramid, 6, video_width, video_height)
                for j in range(len(laplacian_pyramid_resized)):
                    lp = laplacian_pyramid_resized[j]
                    lp = cv2.cvtColor(lp, cv2.COLOR_BGR2RGB) # 
                    lp = transformations_lp(lp)
                    transformed_lp[video_read_index*5+j] = lp

                video_read_index += 1

            frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]
            transformed_lp[i*5,i*5+5] = transformed_lp[-5:]

    video_capture.release()

    video = transformed_video

    return video, transformed_lp, video_name_dis


def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)

        return slow_feature, fast_feature


def video_processing_motion(dist):
    video_name = dist
    video_name_dis = video_name

    video_capture = cv2.VideoCapture()
    video_capture.open(video_name)
    cap = cv2.VideoCapture(video_name)

    video_channel = 3

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_clip = int(video_length / video_frame_rate)

    video_clip_min = 8

    video_length_clip = 32

    transformed_frame_all = torch.zeros([video_length, video_channel, 224, 224])
    transform = transforms.Compose([transforms.Resize([224, 224]), \
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

    transformed_video_all = []

    video_read_index = 0
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_frame = transform(read_frame)
            transformed_frame_all[video_read_index] = read_frame
            video_read_index += 1

    if video_read_index < video_length:
        for i in range(video_read_index, video_length):
            transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

    video_capture.release()

    for i in range(video_clip):
        transformed_video = torch.zeros([video_length_clip, video_channel, 224, 224])
        if (i * video_frame_rate + video_length_clip) <= video_length:
            transformed_video = transformed_frame_all[i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
        else:
            transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
            for j in range((video_length - i * video_frame_rate), video_length_clip):
                transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
        transformed_video_all.append(transformed_video)

    if video_clip < video_clip_min:
        for i in range(video_clip, video_clip_min):
            transformed_video_all.append(transformed_video_all[video_clip - 1])

    return transformed_video_all, video_name_dis


def main(config):
    device = torch.device('cuda' if config.is_gpu else 'cpu')
    print('using ' + str(device))

    model_laplacian = ResNet18_LP().to(device)

    model_motion = slowfast().to(device)


    model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2)

    model = model.to(device=device)
    model = model.float()
    model.load_state_dict(torch.load('ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporal_modular_LSVQ.pth'))

    if config.method_name == 'single-scale':

        video_dist_spatial, video_dist_lp, video_name = video_processing_spatial(config.dist)
        video_dist_motion, video_name = video_processing_motion(config.dist)

        with torch.no_grad():
            model.eval()

            video_dist_spatial = video_dist_spatial.to(device)
            video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)

            video_dist_lp = video_dist_lp.to(device)
            feature_lp = model_laplacian(video_dist_lp).to(device)
            feature_lp = feature_lp.view(8,-1).unsqueeze(0)

            n_clip = len(video_dist_motion)
            feature_motion = torch.zeros([n_clip, 256])


            for idx, ele in enumerate(video_dist_motion):
                ele = ele.unsqueeze(dim=0)
                ele = ele.permute(0, 2, 1, 3, 4)
                ele = pack_pathway_output(ele, device)

                ele_slow_feature, ele_fast_feature = model_motion(ele)

                # ele_slow_feature = ele_slow_feature.squeeze()
                ele_fast_feature = ele_fast_feature.squeeze()

                # ele_feature_motion = torch.cat([ele_slow_feature, ele_fast_feature])
                ele_feature_motion = ele_fast_feature
                ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

                feature_motion[idx] = ele_feature_motion

            feature_motion = feature_motion.unsqueeze(dim=0).to(device)


            y_b, y_s, y_t, y_st = model(video_dist_spatial, feature_motion, feature_lp)

            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--method_name', type=str, default='single-scale')
    parser.add_argument('--dist', type=str, default='/mnt/bn/wenwenwinwin-vqa/LSVQ/yfcc-batch1/746.mp4')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--is_gpu', action='store_false')

    config = parser.parse_args()

    main(config)