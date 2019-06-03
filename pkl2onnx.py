import torch
from core.models.CornerNet_Saccade import model
from core.paths import get_file_path
from core.base import load_cfg, load_nnet
from core.config import SystemConfig


cfg_path = get_file_path("..", "configs", "CornerNet_Saccade.json")
model_path = get_file_path("nnet", "CornerNet_Saccade_500000.pkl")

cfg_sys, cfg_db = load_cfg(cfg_path)
sys_cfg = SystemConfig().update_config(cfg_sys)

cornernet = load_nnet(sys_cfg, model())
example = torch.rand(1, 3, 224, 224).cuda()
torch_out = torch.onnx.export(cornernet.model,
                              example,
                              "test.onnx",
                              verbose=True
                              )
print("onnx done")
