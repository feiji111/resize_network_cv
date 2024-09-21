arch=ResNet_50
result_dir=result/resnet50_imagenet_run3_dataset_partition
dataset_dir=/data/chengqian/cars_classfication
dataset_type=imagenet
backbone_ckpt_path=trained_models/resnet50-0676ba61.pth
device=0,1,4,5
master_port=6681
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node=4 --master_port $master_port main.py \
--phase train \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 8 \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--backbone_ckpt_path $backbone_ckpt_path \
--num_epochs 250 \
--lr 5e-3 \
--warmup_steps 10 \
--warmup_start_lr 4e-5 \
--lr_decay_T_max 250 \
--lr_decay_eta_min 4e-5 \
--weight_decay 2e-5 \
--train_batch_size 256 \
--eval_batch_size 256 \
--ddp \
--apply_resizer_model