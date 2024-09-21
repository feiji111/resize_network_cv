arch=ResNet_50
result_dir=result/run_resnet50_imagenet
dataset_dir=/data/chengqian/cars_classfication
dataset_type=imagenet
model_ckpt_path=/home/zhangkj/resize_network_cv/result/resnet50_imagenet_run3_dataset_partition/model/ResNet_50_best.pt
device=3,4,5,6
master_port=6681
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node=1 --master_port $master_port main.py \
--phase test \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--model_ckpt_path $model_ckpt_path \
--test_batch_size 1 \
--apply_resizer_model