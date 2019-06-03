from .base import Base, load_cfg, load_nnet
from .paths import get_file_path
from .config import SystemConfig
from .dbs.coco import COCO
import torch

class CornerNet(Base):
    def __init__(self):
        from .test.cornernet import cornernet_inference
        from .models.CornerNet import model

        cfg_path   = get_file_path("..", "configs", "CornerNet.json")
        model_path = get_file_path("nnet", "CornerNet_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class CornerNet_Squeeze(Base):
    def __init__(self):
        from .test.cornernet import cornernet_inference
        from .models.CornerNet_Squeeze import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Squeeze.json")
        model_path = get_file_path("nnet", "CornerNet_Squeeze_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_Squeeze, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class CornerNet_Saccade(Base):
    def __init__(self):
        from .test.cornernet_saccade import cornernet_saccade_inference
        from .models.CornerNet_Saccade import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Saccade.json")
        model_path = get_file_path("nnet", "CornerNet_Saccade_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        # print("*"*100)
        # print(cfg_sys, '@@@@@@@@@@@@@@@@@', cfg_db)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        saccade = model()
        # for idx, m in enumerate(saccade.named_modules()):
        #     print(idx, '-->', m)

        cornernet = load_nnet(sys_cfg, saccade)
        """
        example = torch.rand(1, 3, 224, 224).cuda()
        torch_out = torch.onnx.export(cornernet.model,
                                      example,
                                      "test.onnx",
                                      verbose=True
                                      )
        print("onnx done")
        """
        super(CornerNet_Saccade, self).__init__(coco, cornernet, cornernet_saccade_inference, model=model_path)
