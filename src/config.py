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
    'BATCH_SIZE': 4096,
    #'EPOCHS': 20,
    'EPOCHS': 1,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'DATA_PATH': '../data/processed/',
    'OUTPUT_PATH': '../data/output/',
    'MODEL_PATH': 'best_dcn_model.pth'
}

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 시드 고정
seed_everything(CFG['SEED'])
