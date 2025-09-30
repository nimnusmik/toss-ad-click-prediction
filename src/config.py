import os
import random
import torch

def seed_everything(seed):
    """시드 고정 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 설정값들
CFG = {
    'BATCH_SIZE': 4096, #4096->512->1024
    'EPOCHS': 40, # 100
    'FOLDS': 3, #5
    'LEARNING_RATE': 1e-3, #1e-3-> 8e-3->0.001->8e-3-> 0.005
    'SEED': 42,
    'DATA_PATH': '../data/processed/',
    'OUTPUT_PATH': '../data/output/',
    'CHECKPOINT_DIR': '../models/',
    'LOG_DIR': '../logs/',
    'MODEL_PATH': '../models/best_dcn_model.pth',
    'CALIBRATION_PATH': '../models/temperature_calibration.json',
    'USE_WANDB': True,
    'WANDB_PROJECT': 'ctr-dcn-monitoring',
    'WANDB_RUN_NAME': None,
    'WANDB_LOG_EVERY': 1,
    'WANDB_VIZ_EVERY': 5,
    'WANDB_THRESHOLD': 0.5,
}


# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 시드 고정
seed_everything(CFG['SEED'])
