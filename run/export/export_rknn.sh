dataset_dir=/data/chengqian/cars_classfication
onnx_path=/home/zhangkj/resize_network_cv/binary.onnx
rknn_path=binary.rknn
platform=rk3588
dtype=fp
device=3,4,5,6
CUDA_VISIBLE_DEVICES=$device python export.py \
--data $dataset_dir \
--onnx_path $onnx_path \
--rknn_path $rknn_path \
--platform $platform \
--dtype $dtype \
--device cuda \
--rknn_only \
--apply_resizer_model
