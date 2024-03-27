# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import csv
import torch.nn as nn
import random
from data_loader import VideoDataset_images_with_LP_motion_features
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss

from torchvision import transforms
import time
from model import modular


def main(config):
    all_test_SRCC_b, all_test_KRCC_b, all_test_PLCC_b, all_test_RMSE_b = [], [], [], []
    all_test_SRCC_s, all_test_KRCC_s, all_test_PLCC_s, all_test_RMSE_s = [], [], [], []
    all_test_SRCC_t, all_test_KRCC_t, all_test_PLCC_t, all_test_RMSE_t = [], [], [], []
    all_test_SRCC_st, all_test_KRCC_st, all_test_PLCC_st, all_test_RMSE_st = [], [], [], []

    for i in range(10):
        config.exp_version = i
        print('%d round training starts here' % i)
        seed = i * 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
            if config.database == 'Waterloo_indep':
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=9, tr=False, dropout_sp=0.2)
            elif config.database == 'BVISR_indep':
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=5, tr=False, dropout_sp=0.2)
            elif config.database == 'BVIHFR_indep':
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=10, sr=False, dropout_tp=0.2)
            elif config.database == 'LIVEHFR_indep':
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=6, sr=False, dropout_tp=0.2)
            elif config.database == 'Livestreaming_indep':
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=7, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2)
            elif config.database == 'ETRI_indep':
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=5, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2)
            else:
                model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2)

        print('The current model is ' + config.model_name)

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)

        if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
            model = model.float()
        

        if config.trained_model is not None:
            # load the trained model
            print('loading the pretrained model')
            model.load_state_dict(torch.load(config.trained_model))

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
        if config.loss_type == 'plcc':
            criterion = plcc_loss
        elif config.loss_type == 'plcc_rank':
            criterion = plcc_rank_loss
        elif config.loss_type == 'L2':
            criterion = nn.MSELoss().to(device)
        elif config.loss_type == 'L1':
            criterion = nn.L1Loss().to(device)

        elif config.loss_type == 'Huberloss':
            criterion = nn.HuberLoss().to(device)

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        transformations_train = transforms.Compose(  # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  transforms.Resize(config.resize)
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  
             transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC), 
             transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        ## training data
        if config.database == 'LiveVQC':
            videos_dir = 'data/livevqc_image_all_fps1'
            datainfo = 'data/LiveVQC_data.mat'
            lp_dir = 'data/livevqc_LP_ResNet18'

            
            feature_dir = 'data/livevqc_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'LiveVQC_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LiveVQC_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LiveVQC_test', config.crop_size,
                                                                   'Fast', seed=seed)

        elif config.database == 'KoNViD-1k':
            videos_dir = 'data/konvid1k_image_all_fps1'
            datainfo = 'data/KoNViD-1k_data.mat'
            lp_dir = 'data/konvid1k_LP_ResNet18'

            
            feature_dir = 'data/KoNViD-1k_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'KoNViD-1k_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'KoNViD-1k_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'KoNViD-1k_test', config.crop_size,
                                                                   'Fast', seed=seed)

        elif config.database == 'CVD2014':
            videos_dir = 'data/cvd2014_image_all_fps1'
            datainfo = 'data/CVD2014_Realignment_MOS.csv'
            lp_dir = 'data/CVD2014_LP_ResNet18'

            
            feature_dir = 'data/CVD2014_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'CVD2014_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'CVD2014_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'CVD2014_test', config.crop_size,
                                                                   'Fast', seed=seed)

        elif config.database == 'youtube_ugc':
            videos_dir = 'data/youtube_ugc_image_all_fps05'
            datainfo = 'data/youtube_ugc_data.mat'
            lp_dir = 'data/youtube_ugc_LP_ResNet18'

            
            feature_dir = 'data/youtube_ugc_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'youtube_ugc_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'youtube_ugc_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'youtube_ugc_test', config.crop_size,
                                                                   'Fast', seed=seed)

        elif config.database == 'LIVEYTGaming':
            videos_dir = 'data/liveytgaming_image_all_fps1'
            datainfo = 'data/LIVEYTGaming.mat'
            lp_dir = 'data/liveytgaming_LP_ResNet18' #TODO

            
            feature_dir = 'data/LIVEYTGaming_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'LIVEYTGaming_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LIVEYTGaming_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LIVEYTGaming_test', config.crop_size,
                                                                   'Fast', seed=seed)

        elif config.database == 'Livestreaming_indep':
            videos_dir = 'data/livestreaming_image_all_fps1'
            datainfo = 'data/LIVE_livestreaming_scores.csv'
            lp_dir = 'data/livestreaming_rescale_LP_ResNet18' #TODO

            
            feature_dir = 'data/livestreaming_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'Livestreaming_indep_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'Livestreaming_indep_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'Livestreaming_indep_test', config.crop_size,
                                                                   'Fast', seed=seed)
        
        elif config.database == 'ETRI_indep':
            videos_dir = 'data/ETRI_image_all_fps1'
            datainfo = 'data/ETRI_LIVE_MOS.csv'
            lp_dir = 'data/ETRI_LP_ResNet18' 

            
            feature_dir = 'data/ETRI_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'ETRI_indep_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'ETRI_indep_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'ETRI_indep_test', config.crop_size,
                                                                   'Fast', seed=seed)


        elif config.database == 'BVISR_indep':
            videos_dir = 'data/BVI-SR_image_all_fps1'
            datainfo = 'data/BVI-SR_SUB.mat'
            lp_dir = 'data/bvisr_rescale_LP_ResNet18_2' 
            feature_dir = 'data/BVI-SR_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'BVISR_indep_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'BVISR_indep_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'BVISR_indep_test', config.crop_size,
                                                                   'Fast', seed=seed)



        elif config.database == 'Waterloo_indep':
            videos_dir = 'data/waterloo_fps1'
            datainfo = 'data/waterloo_data.csv'
            lp_dir = 'data/waterloo_LP_ResNet18_2' #TODO

            
            feature_dir = 'data/waterloo_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'Waterloo_indep_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'Waterloo_indep_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'Waterloo_indep_test', config.crop_size,
                                                                   'Fast', seed=seed)
        
        elif config.database == 'BVIHFR_indep':
            videos_dir = 'data/BVI-HFR_image_all_fps1'
            datainfo = 'data/BVI-HFR-MOS.csv'
            lp_dir = 'data/BVIHFR_LP_ResNet18' #TODO

            
            feature_dir = 'data/BVIHFR_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'BVIHFR_indep_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'BVIHFR_indep_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'BVIHFR_indep_test', config.crop_size,
                                                                   'Fast', seed=seed)

        elif config.database == 'LIVEHFR_indep':
            videos_dir = 'data/LIVEHFR_image_all_fps1'
            datainfo = 'data/LIVEHFR_MOS.csv'
            lp_dir = 'data/LIVEHFR_LP_ResNet18' #TODO

            
            feature_dir = 'data/LIVEHFR_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'LIVEHFR_indep_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LIVEHFR_indep_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LIVEHFR_indep_test', config.crop_size,
                                                                   'Fast', seed=seed)


        elif config.database == 'LBVD':
            videos_dir = 'data/LBVD_image_all_fps1'
            datainfo = 'data/LBVD_data.mat'
            lp_dir = 'data/LBVD_LP_ResNet18'

            
            feature_dir = 'data/LBVD_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'LBVD_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LBVD_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LBVD_test', config.crop_size,
                                                                   'Fast', seed=seed)


        elif config.database == 'LIVE-Qualcomm':
            videos_dir = 'data/livequalcomm_image_all_fps1'
            datainfo = 'data/LIVE-Qualcomm_qualcommSubjectiveData.mat'
            lp_dir = 'data/livequalcomm_LP_ResNet18' #TODO

            
            feature_dir = 'data/LIVE-Qualcomm_slowfast'
            trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                    transformations_train, 'LIVE-Qualcomm_train', config.crop_size,
                                                                    'Fast', seed=seed)
            valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LIVE-Qualcomm_val', config.crop_size,
                                                                   'Fast', seed=seed)
            testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo,
                                                                   transformations_test, 'LIVE-Qualcomm_test', config.crop_size,
                                                                   'Fast', seed=seed)


        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                 shuffle=False, num_workers=config.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=False, num_workers=config.num_workers)

        best_val_criterion = -1  # SROCC min
        best_val_b, best_val_s, best_val_t, best_val_st = [],[],[],[]
        best_test_b, best_test_s, best_test_t, best_test_st = [],[],[],[]

        print('Starting training:')

        old_save_name = None


        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (video, feature_3D, mos, lp, _) in enumerate(train_loader):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                lp = lp.to(device)
                labels = mos.to(device).float()

                outputs_b, outputs_s, outputs_t, outputs_st = model(video, feature_3D, lp)

                optimizer.zero_grad()

                loss_st = criterion(labels, outputs_st)

                loss = loss_st

                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                loss.backward()

                optimizer.step()

                if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                          (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                           avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))

            # do validation after each epoch
            with torch.no_grad():
                model.eval()

                label = np.zeros([len(valset)])
                y_output_b = np.zeros([len(valset)])
                y_output_s = np.zeros([len(valset)])
                y_output_t = np.zeros([len(valset)])
                y_output_st = np.zeros([len(valset)])
                for i, (video, feature_3D, mos, lp, _) in enumerate(val_loader):
                    video = video.to(device)
                    feature_3D = feature_3D.to(device)
                    lp = lp.to(device)
                    label[i] = mos.item()
                    outputs_b, outputs_s, outputs_t, outputs_st = model(video, feature_3D, lp)

                    y_output_b[i] = outputs_b.item()
                    y_output_s[i] = outputs_s.item()
                    y_output_t[i] = outputs_t.item()
                    y_output_st[i] = outputs_st.item()

                val_PLCC_b, val_SRCC_b, val_KRCC_b, val_RMSE_b = performance_fit(label, y_output_b)
                val_PLCC_s, val_SRCC_s, val_KRCC_s, val_RMSE_s = performance_fit(label, y_output_s)
                val_PLCC_t, val_SRCC_t, val_KRCC_t, val_RMSE_t = performance_fit(label, y_output_t)
                val_PLCC_st, val_SRCC_st, val_KRCC_st, val_RMSE_st = performance_fit(label, y_output_st)

                print(
                    'Epoch {} completed. The result on the base validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        val_SRCC_b, val_KRCC_b, val_PLCC_b, val_RMSE_b))
                print(
                    'Epoch {} completed. The result on the S validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        val_SRCC_s, val_KRCC_s, val_PLCC_s, val_RMSE_s))
                print(
                    'Epoch {} completed. The result on the T validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        val_SRCC_t, val_KRCC_t, val_PLCC_t, val_RMSE_t))
                print(
                    'Epoch {} completed. The result on the ST validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        val_SRCC_st, val_KRCC_st, val_PLCC_st, val_RMSE_st))


                label = np.zeros([len(testset)])
                y_output_b = np.zeros([len(testset)])
                y_output_s = np.zeros([len(testset)])
                y_output_t = np.zeros([len(testset)])
                y_output_st = np.zeros([len(testset)])
                for i, (video, feature_3D, mos, lp, _) in enumerate(test_loader):
                    video = video.to(device)
                    feature_3D = feature_3D.to(device)
                    lp = lp.to(device)
                    label[i] = mos.item()
                    outputs_b, outputs_s, outputs_t, outputs_st = model(video, feature_3D, lp)

                    y_output_b[i] = outputs_b.item()
                    y_output_s[i] = outputs_s.item()
                    y_output_t[i] = outputs_t.item()
                    y_output_st[i] = outputs_st.item()

                test_PLCC_b, test_SRCC_b, test_KRCC_b, test_RMSE_b = performance_fit(label, y_output_b)
                test_PLCC_s, test_SRCC_s, test_KRCC_s, test_RMSE_s = performance_fit(label, y_output_s)
                test_PLCC_t, test_SRCC_t, test_KRCC_t, test_RMSE_t = performance_fit(label, y_output_t)
                test_PLCC_st, test_SRCC_st, test_KRCC_st, test_RMSE_st = performance_fit(label, y_output_st)

                print(
                    'Epoch {} completed. The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b))
                print(
                    'Epoch {} completed. The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s))
                print(
                    'Epoch {} completed. The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t))
                print(
                    'Epoch {} completed. The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                        epoch + 1, \
                        test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st))

                

                if val_SRCC_st > best_val_criterion:
                    print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                    best_val_criterion = val_SRCC_st

                    best_val_b = [val_SRCC_b, val_KRCC_b, val_PLCC_b, val_RMSE_b]
                    best_test_b = [test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b]

                    best_val_s = [val_SRCC_s, val_KRCC_s, val_PLCC_s, val_RMSE_s]
                    best_test_s = [test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s]

                    best_val_t = [val_SRCC_t, val_KRCC_t, val_PLCC_t, val_RMSE_t]
                    best_test_t = [test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t]

                    best_val_st = [val_SRCC_st, val_KRCC_st, val_PLCC_st, val_RMSE_st]
                    best_test_st = [test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st]


                    

        print('Training completed.')

        print(
            'The best training result on the base validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_val_b[0], best_val_b[1], best_val_b[2], best_val_b[3]))
        print(
            'The best training result on the base test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_b[0], best_test_b[1], best_test_b[2], best_test_b[3]))


        print(
            'The best training result on the S validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_val_s[0], best_val_s[1], best_val_s[2], best_val_s[3]))
        print(
            'The best training result on the S test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_s[0], best_test_s[1], best_test_s[2], best_test_s[3]))


        print(
            'The best training result on the T validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_val_t[0], best_val_t[1], best_val_t[2], best_val_t[3]))
        print(
            'The best training result on the T test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_t[0], best_test_t[1], best_test_t[2], best_test_t[3]))


        print(
            'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))
        print(
            'The best training result on the ST test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test_st[0], best_test_st[1], best_test_st[2], best_test_st[3]))


        all_test_SRCC_b.append(best_test_b[0])
        all_test_KRCC_b.append(best_test_b[1])
        all_test_PLCC_b.append(best_test_b[2])
        all_test_RMSE_b.append(best_test_b[3])

        all_test_SRCC_s.append(best_test_s[0])
        all_test_KRCC_s.append(best_test_s[1])
        all_test_PLCC_s.append(best_test_s[2])
        all_test_RMSE_s.append(best_test_s[3])

        all_test_SRCC_t.append(best_test_t[0])
        all_test_KRCC_t.append(best_test_t[1])
        all_test_PLCC_t.append(best_test_t[2])
        all_test_RMSE_t.append(best_test_t[3])

        all_test_SRCC_st.append(best_test_st[0])
        all_test_KRCC_st.append(best_test_st[1])
        all_test_PLCC_st.append(best_test_st[2])
        all_test_RMSE_st.append(best_test_st[3])

        
    print(
        'The base median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(all_test_SRCC_b), np.median(all_test_KRCC_b), np.median(all_test_PLCC_b), np.median(all_test_RMSE_b)))

    print(
        'The S median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(all_test_SRCC_s), np.median(all_test_KRCC_s), np.median(all_test_PLCC_s), np.median(all_test_RMSE_s)))


    print(
        'The T median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(all_test_SRCC_t), np.median(all_test_KRCC_t), np.median(all_test_PLCC_t), np.median(all_test_RMSE_t)))

    print(
        'The ST median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(all_test_SRCC_st), np.median(all_test_KRCC_st), np.median(all_test_PLCC_st), np.median(all_test_RMSE_st)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)

    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)

    # misc
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    parser.add_argument('--loss_type', type=str, default='plcc')

    parser.add_argument('--trained_model', type=str, default='ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporal_modular_LSVQ.pth')


    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)