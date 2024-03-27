# CVPR2024 Modular Blind Video Quality Assessment

## Usage

Prior to commencing, please ensure that you download the relevant datasets and modify the file path accordingly.

Prepare Spatial Rectifier
```python 
python3 ModularBVQA-Demo/extract_laplacian/lp_features.py
```

Prepare Temporal Rectifier
```python 
python3 ModularBVQA/extract_motion/extract_SlowFast_feature_others.py
```

Prepare Base Quality Predictor
```python 
python3 ModularBVQA/extract_frame/extract_frame_konvid1k.py
```

## Model
You can download the trained model via Google Drive: https://drive.google.com/file/d/1WP1DhuUgSOusGc3eqLThTSab2-FMBaJA/view?usp=drive_link.


Finetune and evaluate the model
```python 
python3 -u ModularBVQA/train_other_modular.py \
--database KoNViD-1k \
--model_name ViTbCLIP_SpatialTemporal_modular_dropout \
--conv_base_lr 1e-5 \
--epochs 30 \
--train_batch_size 16 \
--print_samples 1000 \
--num_workers 16 \
--resize 224 \
--crop_size 224 \
--decay_ratio 0.9 \
--decay_interval 2 \
--loss_type plcc \
--trained_model ModularBVQA-Demo/ckpts_modular/ViTbCLIP_SpatialTemporal_modular_LSVQ.pth
```


Evaluate the model pretrained on LSVQ
```python 
python3 -u ModularBVQA/test_baseline_modular.py \
--database KoNViD-1k \
--train_database LSVQ \
--model_name ViTbCLIP_SpatialTemporal_modular_dropout \
--trained_model ModularBVQA/ckpts_modular/ViTbCLIP_SpatialTemporal_modular_LSVQ.pth \
--num_workers 16 \
--data_path / \
--resize 224 \
--crop_size 224
```

