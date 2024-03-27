import argparse
import numpy as np

import torch
import torch.nn
from torchvision import transforms
from model import modular
from utils import performance_fit
from data_loader import VideoDataset_images_with_LP_motion_features


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=8, sr=True, tr=True)
    # config.multi_gpu = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)


    if config.model_name == 'ViTbCLIP_SpatialTemporal_modular_dropout':
        model = model.float()

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))


    ## training data
    if config.database == 'LiveVQC':
        videos_dir = 'data/livevqc_image_all_fps1'
        datainfo = 'data/LiveVQC_data.mat'
        lp_dir = 'data/livevqc_LP_ResNet18'
        feature_dir = 'data/livevqc_slowfast'

    elif config.database == 'KoNViD-1k':
        videos_dir = 'data/konvid1k_image_all_fps1'
        datainfo = 'data/KoNViD-1k_data.mat'
        lp_dir = 'data/konvid1k_LP_ResNet18'     
        feature_dir = 'data/KoNViD-1k_slowfast'

    elif config.database == 'CVD2014':
        videos_dir = 'data/cvd2014_image_all_fps1'
        datainfo = 'data/CVD2014_Realignment_MOS.csv'
        lp_dir = 'data/CVD2014_LP_ResNet18'    
        feature_dir = 'data/CVD2014_slowfast'

    elif config.database == 'youtube_ugc':
        videos_dir = 'data/youtube_ugc_image_all_fps05'
        datainfo = 'data/youtube_ugc_data.mat'
        lp_dir = 'data/youtube_ugc_LP_ResNet18'    
        feature_dir = 'data/youtube_ugc_slowfast'

    elif config.database == 'LIVEYTGaming':
        videos_dir = 'data/liveytgaming_image_all_fps1'
        datainfo = 'data/LIVEYTGaming.mat'
        lp_dir = 'data/liveytgaming_LP_ResNet18' 
        feature_dir = 'data/LIVEYTGaming_slowfast'


    elif config.database == 'LBVD':
        videos_dir = 'data/LBVD_image_all_fps1'
        datainfo = 'data/LBVD_data.mat'
        lp_dir = 'data/LBVD_LP_ResNet18'     
        feature_dir = 'data/LBVD_slowfast'


    elif config.database == 'LIVE-Qualcomm':
        videos_dir = 'data/livequalcomm_image_all_fps1'
        datainfo = 'data/LIVE-Qualcomm_qualcommSubjectiveData.mat'
        lp_dir = 'data/livequalcomm_LP_ResNet18' 
        feature_dir = 'data/LIVE-Qualcomm_slowfast'


    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC), #transforms.Resize(config.resize), 
         transforms.CenterCrop(config.crop_size), 
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = VideoDataset_images_with_LP_motion_features(videos_dir, feature_dir, lp_dir, datainfo, transformations_test, config.database, config.crop_size, 'Fast')

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)


    with torch.no_grad():
        model.eval()

        label = np.zeros([len(testset)])
        y_output_b = np.zeros([len(testset)])
        y_output_s = np.zeros([len(testset)])
        y_output_t = np.zeros([len(testset)])
        y_output_st = np.zeros([len(testset)])
        for i, (video, feature_3D, mos, lp, video_name) in enumerate(test_loader):
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

        print(config.database)
        print('The result on the base test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_b, test_KRCC_b, test_PLCC_b, test_RMSE_b))
        print('The result on the S test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_s, test_KRCC_s, test_PLCC_s, test_RMSE_s))
        print('The result on the T test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(test_SRCC_t, test_KRCC_t, test_PLCC_t, test_RMSE_t))
        print('The result on the ST test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( test_SRCC_st, test_KRCC_st, test_PLCC_st, test_RMSE_st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='youtube_ugc') 
    parser.add_argument('--train_database', type=str, default='LSVQ')
    parser.add_argument('--model_name', type=str, default='ViTbCLIP_SpatialTemporal_modular_dropout') 

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str, default='ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporal_modular_LSVQ.pth') 
    parser.add_argument('--data_path', type=str, default='/')
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--resize', type=int, default=224) 
    parser.add_argument('--crop_size', type=int, default=224) 

    config = parser.parse_args()

    main(config)