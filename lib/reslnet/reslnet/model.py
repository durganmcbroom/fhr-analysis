from torch import nn
from transformers import ResNetConfig


class ResLNet(nn.Module):
    def __init__(
            self,
            res_config: ResNetConfig
    ):
        super().__init__()

