import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils, loss, meter, scheduler
from data.dataset import Dataset_cifar10, Dataset_cifar100, Dataset_imagenet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model.model import ResNetResizer

import pdb
from PIL import Image
import matplotlib.pyplot as plt

class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.model_ckpt_path = args.model_ckpt_path

    def dataload(self):
        image_size = 416
        test_dir = os.path.join(self.dataset_dir, "test")

        testset = ImageFolder(
            test_dir,
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)
                    ),
                ]
            ),
        )

        self.test_loader = DataLoader(
                testset,
                batch_size=self.test_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
        )

        print("Dataset has been loaded!")

        classes = testset.classes

        class_to_idx = testset.class_to_idx

        print("Classes:", classes)
        print("Class to index mapping:", class_to_idx)

    def build_model(self):
        print("==> Building model..")

        print("Loading student model")
        self.model = ResNetResizer(self.args)
        ckpt_model = torch.load(self.model_ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt_model['model'])

    def test(self):
        if self.device == "cuda":
            self.model = self.model.cuda()

        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        self.model.eval()
        # a = 1
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), ncols=100) as _tqdm:
                for images, targets in self.test_loader:
                    # to_pil = transforms.ToPILImage()
                    # tmp_images = to_pil(images.squeeze())
                    # tmp_images.save("/home/zhangkj/resize_network_cv/tmp/" + str(a) + ".jpg")
                    # a = a + 1

                    # pdb.set_trace()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits = self.model(images)
                    prec1 = utils.get_accuracy(
                        logits, targets, topk=(1,)
                    )
                    n = images.size(0)
                    meter_top1.update(prec1[0].item(), n)

                    _tqdm.set_postfix(
                        top1="{:.4f}".format(meter_top1.avg),
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

        print(
            "[Test] "
            "Prec@(1) {top1:.2f}".format(
                top1=meter_top1.avg,
            )
        )

    def main(self):
        self.dataload()
        self.build_model()
        self.test()