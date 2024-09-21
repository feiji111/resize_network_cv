from .backbone import get_base_model
from .resizer import Resizer
import torch.nn as nn

def get_model(name, args):
    if name == "resizer":
        return Resizer(args)
    elif name == "backbone":
        if args.apply_resizer_model:
            in_channels = args.out_channels
        else:
            in_channels = args.in_channels
        return get_base_model(in_channels, args.num_classes, args.pretrained)
    else:
        raise ValueError(f"Incorrect name={name}. The valid options are"
                         "('resizer', 'backbone')")

class ResNetResizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.apply_resizer_model:
            self.resizer = get_model('resizer', args)
        else:
            self.resizer = None

        self.base_model = get_model('backbone', args)

    def forward(self, x):
        if self.resizer is not None:
            x = self.resizer(x)
        x = self.base_model(x)
        return x