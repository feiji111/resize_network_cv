import argparse
import torch
import sys
import os
import platform
from pathlib import Path
from model.model import ResNetResizer

from rknn.api import RKNN

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_args():
    desc = "Transform Resize Network To ONNX"
    parser = argparse.ArgumentParser(description=desc)
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
    parser.add_argument("--interpolate_mode",
        default="bilinear",
        type=str,
        help="Passed to torch.nn.functional.interpolate"
    )
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--ckpt_path", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[416, 416], help="image (h, w)")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--per-tensor", action="store_true", help="TF per-tensor quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--onnx_path", type=str, default=ROOT / "binary.onnx", help="onnx model path")
    parser.add_argument("--rknn_path", type=str, default=ROOT / "binary.rknn", help="rknn model path")
    parser.add_argument(
        "--platform", 
        type=str,
        default="rk3588",
        choices=("rk3588", "rk3566", "rk3568", "rk3562"),
        help="platform to deploy"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp",
        choices=("fp", "i8"),
        help="quantization or not"
    )
    parser.add_argument("--rknn_only", action="store_true", help="Export to rknn model")
    parser.add_argument("--onnx_only", action="store_true", help="Export to onnx model")
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
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")

    return parser.parse_args()

class ExportONNX:
    def __init__(self, args):
        self.args = args
        self.data = args.data
        self.device = args.device
        self.batch_size = args.batch_size
        self.ckpt_path = args.ckpt_path
        self.onnx_path = args.onnx_path
        self.simplify = args.simplify
    
    def build_model(self):
        print("==> Building model..")

        print("Loading student model")
        self.model = ResNetResizer(self.args)
        ckpt_model = torch.load(self.ckpt_path[0], map_location="cpu")
        self.model.load_state_dict(ckpt_model['model'])

    def convert(self):
        batch_size = 1
        input_shape = (3, 416, 416)
        input = torch.randn(batch_size, *input_shape)

        torch.onnx.export(
            self.model,
            input,
            self.onnx_path,
            input_names=['input'],
            output_names=['output'],
            keep_initializers_as_inputs=True,
            verbose=True
        )

        # if self.simplify:
        #     try:
        #         cuda = torch.cuda.is_available()
        #         check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1"))
        #         import onnxsim

        #         LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
        #         model_onnx, check = onnxsim.simplify(model_onnx)
        #         assert check, "assert check failed"
        #         onnx.save(model_onnx, f)
        #     except Exception as e:
        #         LOGGER.info(f"{prefix} simplifier failure: {e}")

    def main(self):
        self.build_model()
        self.convert()

class ExportRKNN:
    def __init__(self, args):
        self.args = args
        self.data = args.data
        self.device = args.device
        self.btch_size = args.batch_size
        self.onnx_path = args.onnx_path
        self.rknn_path = args.rknn_path
        self.platform = args.platform
        self.dtype = args.dtype
    
    def build_model(self):
        self.rknn = RKNN(verbose = True)
        self.rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=self.platform)
        self.rknn.load_onnx(self.onnx_path)

    def convert(self):
        do_quant = False
        if self.dtype == "i8":
            do_quant = True
        else:
            do_quant = False

        self.rknn.build(do_quant, self.data)
        self.rknn.export_rknn(self.rknn_path)
    
    def main(self):
        self.build_model()
        self.convert()
        self.rknn.release()

def main():
    args = parse_args()
    if args.onnx_only:
        exportONNX = ExportONNX(args)
        exportONNX.main()
    elif args.rknn_only:
        exportRKNN = ExportRKNN(args)
        exportRKNN.main()
    else: 
        exportONNX = ExportONNX(args)
        exportRKNN = ExportRKNN(args)
        exportONNX.main()
        exportRKNN.main()

if __name__ == "__main__":
    main()


