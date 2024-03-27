CUDA_VISIBLE_DEVICES=0 python3 -u ModularBVQA-Demo/train_other_modular.py \
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

