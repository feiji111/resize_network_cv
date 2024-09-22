dataset_dir=/data/chengqian/cars_classfication
weights_dir=/home/zhangkj/resize_network_cv/result/resnet50_imagenet_run3_dataset_partition/model/ResNet_50_best.pt
device=3,4,5,6
CUDA_VISIBLE_DEVICES=$device python export.py \
--data $dataset_dir \
--ckpt_path $weights_dir \
--device cuda \
--onnx_only \
--apply_resizer_model
