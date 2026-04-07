import torch
import torch.nn as nn
from safetensors.torch import safe_open
from xtuner.model.titan.TITAN_local.modeling_titan import Titan
from xtuner.model.titan.TITAN_local.configuration_titan import TitanConfig

class TitanVisionTower(nn.Module):
    def __init__(self, config_path, weight_path):
        super().__init__()
        self.titan = self.make_titan_from_local(config_path, weight_path)

    def make_titan_from_local(self, config_path, weight_path):
        config = TitanConfig.from_json_file(config_path)
        model = Titan(config)
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(state_dict)
        return model

    def forward(self, features, coords, patch_size_lv0):
        return self.titan.get_visual_final_features(features, coords, patch_size_lv0)
