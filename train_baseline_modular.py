# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from data_loader import VideoDataset_images_with_LP_motion_features
from utils import performance_fit
from utils import plcc_loss, plcc_rank_loss, plcc_l1_loss
from model import modular

from torchvision import transforms
import time


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2)

    print('The current model is ' + config.model_name)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    if config.model_name ==  'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = model.float()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'plcc':
        criterion = plcc_loss
    elif config.loss_type == 'plcc_l1':
        criterion = plcc_l1_loss
    elif config.loss_type == 'plcc_rank':
        criterion = plcc_rank_loss
    elif config.loss_type == 'l2':
        criterion = nn.MSELoss().to(device)
    elif config.loss_type == 'l1':
        criterion = nn.L1Loss().to(device)

    elif config.loss_type == 'Huberloss':
        criterion = nn.HuberLoss().to(device)

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    ## training data
    if config.database == 'LSVQ':
        videos_dir = 'data/LSVQ_image_all_fps1_motion'
        datainfo_train = 'data/LSVQ_whole_train.csv'
        datainfo_val = 'data/LSVQ_whole_train.csv'
        datainfo_test = 'data/LSVQ_whole_test.csv'
        datainfo_test_1080p = 'data/LSVQ_whole_test_1080p.csv'

        transformations_train = transforms.Compose(  # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  BILINEAR NEAREST
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),  
             transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose(
            [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC), 
             transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



        feature_dir = 'data/LSVQ_slowfast'
        lp_dir = 'data/LSVQ_LP_ResNet18'  
        trainset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo_train,
                                                                transformations_train, 'LSVQ_train',
                                                                config.crop_size,
                                                                'Fast')
        valset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo_val,
                                                              transformations_test, 'LSVQ_val', config.crop_size,
                                                              'Fast')
        testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo_test,
                                                               transformations_test, 'LSVQ_test', config.crop_size,
                                                               'Fast')
        testset_1080p = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo_test_1080p,
                                                                     transformations_test, 'LSVQ_test_1080p',
                                                                     config.crop_size, 'Fast')


            ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                             shuffle=False, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1,
                                                    shuffle=False, num_workers=config.num_workers)

    best_val_criterion = -1  # SROCC min
    best_val_b, best_val_s, best_val_t, best_val_st = [],[],[],[]
    best_test_b, best_test_s, best_test_t, best_test_st = [],[],[],[]
    best_test_b_1080p, best_test_s_1080p, best_test_t_1080p, best_test_st_1080p = [],[],[],[]

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

            label_1080p = np.zeros([len(testset_1080p)])
            y_output_b_1080p = np.zeros([len(testset_1080p)])
            y_output_s_1080p = np.zeros([len(testset_1080p)])
            y_output_t_1080p = np.zeros([len(testset_1080p)])
            y_output_st_1080p = np.zeros([len(testset_1080p)])

            for i, (video, feature_3D, mos, lp, _) in enumerate(test_loader_1080p):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                lp = lp.to(device)
                label_1080p[i] = mos.item()
                outputs_b_1080p, outputs_s_1080p, outputs_t_1080p, outputs_st_1080p = model(video, feature_3D, lp)

                y_output_b_1080p[i] = outputs_b_1080p.item()
                y_output_s_1080p[i] = outputs_s_1080p.item()
                y_output_t_1080p[i] = outputs_t_1080p.item()
                y_output_st_1080p[i] = outputs_st_1080p.item()

            test_PLCC_b_1080p, test_SRCC_b_1080p, test_KRCC_b_1080p, test_RMSE_b_1080p = performance_fit(label_1080p, y_output_b_1080p)
            test_PLCC_s_1080p, test_SRCC_s_1080p, test_KRCC_s_1080p, test_RMSE_s_1080p = performance_fit(label_1080p, y_output_s_1080p)
            test_PLCC_t_1080p, test_SRCC_t_1080p, test_KRCC_t_1080p, test_RMSE_t_1080p = performance_fit(label_1080p, y_output_t_1080p)
            test_PLCC_st_1080p, test_SRCC_st_1080p, test_KRCC_st_1080p, test_RMSE_st_1080p = performance_fit(label_1080p, y_output_st_1080p)

            print(
                'Epoch {} completed. The result on the base test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, \
                    test_SRCC_b_1080p, test_KRCC_b_1080p, test_PLCC_b_1080p, test_RMSE_b_1080p))
            print(
                'Epoch {} completed. The result on the S test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, \
                    test_SRCC_s_1080p, test_KRCC_s_1080p, test_PLCC_s_1080p, test_RMSE_s_1080p))
            print(
                'Epoch {} completed. The result on the T test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, \
                    test_SRCC_t_1080p, test_KRCC_t_1080p, test_PLCC_t_1080p, test_RMSE_t_1080p))
            print(
                'Epoch {} completed. The result on the ST test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, \
                    test_SRCC_st_1080p, test_KRCC_st_1080p, test_PLCC_st_1080p, test_RMSE_st_1080p))

            if val_SRCC_st > best_val_criterion:
                print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                best_val_criterion = val_SRCC_st

                best_val_b = [val_SRCC_b, val_KRCC_b, val_PLCC_b, val_RMSE_b]
                best_test_b = [test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b]
                best_test_b_1080p = [test_SRCC_b_1080p, test_KRCC_b_1080p, test_PLCC_b_1080p, test_RMSE_b_1080p]

                best_val_s = [val_SRCC_s, val_KRCC_s, val_PLCC_s, val_RMSE_s]
                best_test_s = [test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s]
                best_test_s_1080p = [test_SRCC_s_1080p, test_KRCC_s_1080p, test_PLCC_s_1080p, test_RMSE_s_1080p]

                best_val_t = [val_SRCC_t, val_KRCC_t, val_PLCC_t, val_RMSE_t]
                best_test_t = [test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t]
                best_test_t_1080p = [test_SRCC_t_1080p, test_KRCC_t_1080p, test_PLCC_t_1080p, test_RMSE_t_1080p]

                best_val_st = [val_SRCC_st, val_KRCC_st, val_PLCC_st, val_RMSE_st]
                best_test_st = [test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st]
                best_test_st_1080p = [test_SRCC_st_1080p, test_KRCC_st_1080p, test_PLCC_st_1080p, test_RMSE_st_1080p]

                # print('Saving model...')
                # if not os.path.exists(config.ckpt_path):
                #     os.makedirs(config.ckpt_path)

                # if epoch > 0:
                #     if os.path.exists(old_save_name):
                #         os.remove(old_save_name)

                # save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                #                                config.database + '_' + config.loss_type + '_NR_v' + str(
                #     config.exp_version) \
                #                                + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC_st))
                # torch.save(model.state_dict(), save_model_name)
                # old_save_name = save_model_name

    print('Training completed.')
    print(
        'The best training result on the base validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_val_b[0], best_val_b[1], best_val_b[2], best_val_b[3]))
    print(
        'The best training result on the base test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_b[0], best_test_b[1], best_test_b[2], best_test_b[3]))
    print(
        'The best training result on the base test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_b_1080p[0], best_test_b_1080p[1], best_test_b_1080p[2], best_test_b_1080p[3]))

    print(
        'The best training result on the S validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_val_s[0], best_val_s[1], best_val_s[2], best_val_s[3]))
    print(
        'The best training result on the S test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_s[0], best_test_s[1], best_test_s[2], best_test_s[3]))
    print(
        'The best training result on the S test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_s_1080p[0], best_test_s_1080p[1], best_test_s_1080p[2], best_test_s_1080p[3]))

    print(
        'The best training result on the T validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_val_t[0], best_val_t[1], best_val_t[2], best_val_t[3]))
    print(
        'The best training result on the T test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_t[0], best_test_t[1], best_test_t[2], best_test_t[3]))
    print(
        'The best training result on the T test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_t_1080p[0], best_test_t_1080p[1], best_test_t_1080p[2], best_test_t_1080p[3]))

    print(
        'The best training result on the ST validation dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_val_st[0], best_val_st[1], best_val_st[2], best_val_st[3]))
    print(
        'The best training result on the ST test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_st[0], best_test_st[1], best_test_st[2], best_test_st[3]))
    print(
        'The best training result on the ST test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test_st_1080p[0], best_test_st_1080p[1], best_test_st_1080p[2], best_test_st_1080p[3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LSVQ')
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
    parser.add_argument('--ckpt_path', type=str, default='ModularBVQA/ckpts_modular')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    parser.add_argument('--loss_type', type=str, default='plcc')


    config = parser.parse_args()

    torch.manual_seed(0)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    main(config)