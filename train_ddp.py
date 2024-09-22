import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import utils, meter, scheduler, loss
from model.model import ResNetResizer
from data.dataset import Dataset_imagenet


class TrainDDP:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.resume = args.resume

        self.start_epoch = 0
        self.best_prec1 = 0

        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

    def dist_init(self):
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

    def result_init(self):
        # tensorboard
        if self.rank == 0:
            self.writer = SummaryWriter(self.result_dir)

            # log
            self.logger = utils.get_logger(
                os.path.join(self.result_dir, "train_logger.log"), "train_logger"
            )

            # config
            self.logger.info("train config:")
            self.logger.info(str(json.dumps(vars(self.args), indent=4)))
            utils.record_config(
                self.args, os.path.join(self.result_dir, "train_config.txt")
            )

            self.logger.info("--------- Train -----------")

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(False)

        # avoid homogenization
        self.seed = self.seed + self.rank

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        dataset = eval("Dataset_" + self.dataset_type)(
                self.args.image_size,
                self.dataset_dir,
                self.train_batch_size,
                self.eval_batch_size,
                self.num_workers,
                self.pin_memory,
                ddp=True,
            )
        self.train_loader, self.val_loader = (
            dataset.loader_train,
            dataset.loader_test,
        )
        if self.rank == 0:
            self.logger.info("Dataset has been loaded!")

    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")

            self.logger.info("Loading model")
        self.model = ResNetResizer(self.args)
        ckpt_backbone = torch.load(self.args.backbone_ckpt_path, map_location="cpu")
        # print(ckpt_backbone["fc.weight"].shape)
        self.model.load_state_dict(ckpt_backbone, strict=False)
        print(self.model.base_model.state_dict()['fc.weight'].shape)

    def define_loss(self):
        self.ori_loss = nn.CrossEntropyLoss()

    def define_optim(self):
        # split weight and mask
        weight_params = self.model.parameters()

        # optim
        self.optim_weight = torch.optim.Adamax(
            weight_params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-7
        )

        # scheduler
        self.scheduler_model_weight = scheduler.CosineAnnealingLRWarmup(
            self.optim_weight,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )
    
    def resume_model_ckpt(self):
        ckpt_model = torch.load(self.resume)

        self.best_prec1 = ckpt_model["best_prec1"]
        self.start_epoch = ckpt_model["start_epoch"]
        self.model.load_state_dict(ckpt_model["model"])
        self.optim_weight.load_state_dict(ckpt_model["optim_weight"])
        self.scheduler_model_weight.load_state_dict(
            ckpt_model["scheduler_model_weight"]
        )
        if self.rank == 0:
            self.logger.info("=> Continue from epoch {}...".format(self.start_epoch))

    def save_model_ckpt(self, is_best):
        folder = os.path.join(self.result_dir, "model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt_model = {}
        ckpt_model["best_prec1"] = self.best_prec1
        ckpt_model["start_epoch"] = self.start_epoch
        ckpt_model["model"] = self.model.module.state_dict()
        ckpt_model["optim_weight"] = self.optim_weight.state_dict()
        ckpt_model[
            "scheduler_model_weight"
        ] = self.scheduler_model_weight.state_dict()

        if is_best:
            torch.save(
                ckpt_model,
                os.path.join(folder, self.arch + "_best.pt"),
            )
        torch.save(ckpt_model, os.path.join(folder, self.arch + "_last.pt"))


    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    
    def train(self):
        self.model = self.model.cuda()
        self.ori_loss = self.ori_loss.cuda()

        if self.resume:
            self.resume_model_ckpt()

        self.model = DDP(self.model, find_unused_parameters=False)

        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)
            # train
            self.model.train()
            # self.model.module.ticket = False

            if self.rank == 0:
                meter_oriloss.reset()

                meter_loss.reset()
                meter_top1.reset()
                lr = (
                    self.optim_weight.state_dict()["param_groups"][0]["lr"]
                    if epoch > 1
                    else self.warmup_start_lr
                )

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                for data in self.train_loader:
                    self.optim_weight.zero_grad()

                    images = data[0].cuda()
                    targets = data[1].cuda()
                    
                    logits = self.model(images)

                    # loss
                    ori_loss = self.ori_loss(logits, targets)

                    # Flops = self.model.module.get_flops()

                    total_loss = (
                        ori_loss
                    )

                    total_loss.backward()
                    self.optim_weight.step()

                    prec1, = utils.get_accuracy(
                        logits, targets, topk=(1, )
                    )

                    dist.barrier()
                    # reduced_ori_loss = self.reduce_tensor(ori_loss)
                    # reduced_total_loss = self.reduce_tensor(total_loss)
                    # reduced_prec1 = self.reduce_tensor(prec1)
                    if self.rank == 0:
                        reduced_ori_loss = ori_loss
                        reduced_total_loss = total_loss
                        reduced_prec1 = prec1

                    if self.rank == 0:
                        n = images.size(0)
                        meter_oriloss.update(reduced_ori_loss.item(), n)
                        meter_loss.update(reduced_total_loss.item(), n)
                        meter_top1.update(reduced_prec1.item(), n)

                        _tqdm.set_postfix(
                            loss="{:.4f}".format(meter_loss.avg),
                            top1="{:.4f}".format(meter_top1.avg),
                        )
                        _tqdm.update(1)

                    time.sleep(0.01)

            self.scheduler_model_weight.step()

            if self.rank == 0:
                # Flops = self.student.module.get_flops()
                self.writer.add_scalar(
                    "train/loss/ori_loss",
                    meter_oriloss.avg,
                    global_step=epoch,
                )
                self.writer.add_scalar(
                    "train/loss/total_loss",
                    meter_loss.avg,
                    global_step=epoch,
                )

                self.writer.add_scalar(
                    "train/acc/top1",
                    meter_top1.avg,
                    global_step=epoch,
                )

                self.writer.add_scalar(
                    "train/lr/lr",
                    lr,
                    global_step=epoch,
                )
                # self.writer.add_scalar(
                #     "train/Flops",
                #     Flops,
                #     global_step=epoch,
                # )

                self.logger.info(
                    "[Train] "
                    "Epoch {0} : "
                    "LR {lr:.6f} "
                    "OriLoss {ori_loss:.4f} "
                    "TotalLoss {total_loss:.4f} "
                    "Prec@(1) {top1:.2f}".format(
                        epoch,
                        lr=lr,
                        ori_loss=meter_oriloss.avg,
                        total_loss=meter_loss.avg,
                        top1=meter_top1.avg,
                    )
                )
                # self.logger.info(
                #     "[Train model Flops] Epoch {0} : ".format(epoch)
                #     + str(Flops.item() / (10**6))
                #     + "M"
                # )

            # valid
            if self.rank == 0:
                self.model.eval()
                # self.model.module.ticket = True
                meter_top1.reset()
                with torch.no_grad():
                    with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                        _tqdm.set_description(
                            "epoch: {}/{}".format(epoch, self.num_epochs)
                        )
                        for images, targets in self.val_loader:
                            images = images.cuda()
                            targets = targets.cuda()
                            logits = self.model(images)
                            prec1 = utils.get_accuracy(
                                logits, targets, topk=(1, )
                            )[0]
                            n = images.size(0)
                            meter_top1.update(prec1.item(), n)

                            _tqdm.set_postfix(
                                top1="{:.4f}".format(meter_top1.avg),
                            )
                            _tqdm.update(1)
                            time.sleep(0.01)

                self.writer.add_scalar(
                    "val/acc/top1",
                    meter_top1.avg,
                    global_step=epoch,
                )

                self.logger.info(
                    "[Val] "
                    "Epoch {0} : "
                    "Prec@(1) {top1:.2f}".format(
                        epoch,
                        top1=meter_top1.avg,
                    )
                )

                self.start_epoch += 1
                if self.best_prec1 < meter_top1.avg:
                    self.best_prec1 = meter_top1.avg
                    self.save_model_ckpt(True)
                else:
                    self.save_model_ckpt(False)

                self.logger.info(
                    " => Best top1 accuracy before finetune : " + str(self.best_prec1)
                )
        if self.rank == 0:
            self.logger.info("Trian finished!")

    def main(self):
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()