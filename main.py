import argparse
import time
# from train import Train
from train_ddp import TrainDDP
from test import Test
import torch.distributed as dist

def parse_args():
    desc = "Pytorch implementation of Resize Network"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=("train", "finetune", "test"),
        help="train, finetune or test",
    )

    # common
    parser.add_argument(
        "--dataset_dir", type=str, default="./data", help="The dataset path"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar10",
        choices=("cifar10", "cifar100", "imagenet"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The num_workers of dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="The pin_memory of dataloader",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_18",
        choices=(
            "ResNet_18",
            "ResNet_50",
            "VGG_16_bn",
            "resnet_56",
            "resnet_110",
            "DenseNet_40",
            "GoogLeNet",
            "MobileNetV2",
        ),
        help="The architecture to backbone",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default="3407", help="Init seed")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./result/",
        help="The directory where the results will be stored",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use the distributed data parallel",
    )

    # train
    parser.add_argument(
        "--backbone_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the pretrained backbone ckpt",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=350, help="The num of epochs to train."
    )
    parser.add_argument(
        "--lr", default=5e-4, type=float, help="The initial learning rate of model"
    )
    parser.add_argument(
        "--warmup_steps",
        default=30,
        type=int,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--warmup_start_lr",
        default=1e-4,
        type=float,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--lr_decay_T_max",
        default=350,
        type=int,
        help="T_max of CosineAnnealingLR",
    )
    parser.add_argument(
        "--lr_decay_eta_min",
        default=5e-6,
        type=float,
        help="eta_min of CosineAnnealingLR",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--train_batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=256, help="Batch size for validation"
    )
    parser.add_argument(
        "--coef_kdloss", type=float, default=0.5, help="Coefficient of kd loss"
    )
    parser.add_argument(
        "--coef_rcloss",
        type=float,
        default=100,
        help="Coefficient of reconstruction loss",
    )
    parser.add_argument(
        "--coef_maskloss", type=float, default=1.0, help="Coefficient of mask loss"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="load the model from the specified checkpoint",
    )
    parser.add_argument(
        "--apply_resizer_model",
        action="store_true",
        help="use resizer network"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use pretrained backbone"
    )
    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
        help="num of classes"
    )
    parser.add_argument(
        "--resizer_image_size",
        default=416,
        type=int,
        help="size of images passed to resizer model"
    )
    parser.add_argument(
        "--image_size",
        default=32,
        type=int,
        help="size of images passed to CNN model"
    )
    parser.add_argument(
        "--in_channels",
        default=3,
        type=int,
        help="Number of input channels of resizer (for RGB images it is 3)"
    )
    parser.add_argument(
        "--out_channels",
        default=3,
        type=int,
        help="Number of output channels of resizer (for RGB images it is 3)"
    )
    parser.add_argument(
        "--num_kernels",
        default=16,
        type=int,
        help="Same as `n` in paper 16 original"
    )
    parser.add_argument(
        "--num_resblocks",
        default=2,
        type=int,
        help="Same as 'r' in paper 2 original"
    )
    parser.add_argument(
        "--negative_slope",
        default=0.2,
        type=float,
        help="Used by leaky relu"
    )
    parser.add_argument(
        "--interpolate_mode",
        default="bilinear",
        type=str,
        help="Passed to torch.nn.functional.interpolate"
    )

    # finetune
    parser.add_argument(
        "--finetune_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the student ckpt in finetune",
    )
    parser.add_argument(
        "--finetune_num_epochs",
        type=int,
        default=100,
        help="The num of epochs to train in finetune",
    )
    parser.add_argument(
        "--finetune_lr",
        default=1e-5,
        type=float,
        help="The initial learning rate of model in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_steps",
        default=10,
        type=int,
        help="The steps of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_start_lr",
        default=1e-4,
        type=float,
        help="The steps of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_T_max",
        default=100,
        type=int,
        help="T_max of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_eta_min",
        default=5e-6,
        type=float,
        help="eta_min of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay in finetune",
    )
    parser.add_argument(
        "--finetune_train_batch_size",
        type=int,
        default=256,
        help="Batch size for training in finetune",
    )
    parser.add_argument(
        "--finetune_eval_batch_size",
        type=int,
        default=256,
        help="Batch size for validation in finetune",
    )
    parser.add_argument(
        "--finetune_resume",
        type=str,
        default=None,
        help="load the model from the specified checkpoint in finetune",
    )

    # test
    parser.add_argument(
        "--test_batch_size", type=int, default=256, help="Batch size for test"
    )

    # get_flops_and_params
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the sparsed student ckpt",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    if args.ddp:
        if args.phase == "train":
            train = TrainDDP(args=args)
            train.main()
        # elif args.phase == "finetune":
        #     finetune = FinetuneDDP(args=args)
        #     finetune.main()
    else:
        if args.phase == "train":
            # train = Train(args=args)
            train.main()
        # elif args.phase == "finetune":
        #     finetune = Finetune(args=args)
        #     finetune.main()
        elif args.phase == "test":
            test = Test(args=args)
            test.main()


if __name__ == "__main__":
    main()