from typing import List, Callable, Optional, Any, cast, Dict

import os
import gym
import torch
import inspect
import importlib
import numpy as np
from torch import nn as nn
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.main import find_sub_modules

class IdentityPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        input_height: int,
        input_width: int,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any
    ):
        def f(x, k):
            assert k in x, "{} must be set in CustomPreprocessor".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height = input_height
        self.input_width = input_width
        self.shape = (input_width, input_height)

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        self.identity_model = nn.Sequential(nn.Identity())
        self.model = self.identity_model.to(self.device)
        
        low = -np.inf
        high = np.inf

        assert (
            len(input_uuids) == 1
        ), "custom preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=self.shape)

        super().__init__(**prepare_locals_for_super(locals()))

    def to(self, device: torch.device) -> str:
        self.model = self.model.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        # print(obs)
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        return self.model(x.to(self.device))
