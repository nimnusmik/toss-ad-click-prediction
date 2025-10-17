import os
import random
import torch
import numpy as np

# 1. 시드 고정 함수
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

# 2. DataLoader worker 시드 고정
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    'CALIBRATION_ENABLED': True,
    'CALIBRATION_HOLDOUT_FRACTION': 0.1,
    'CALIBRATION_MAX_SAMPLES': 250000,
    'CALIBRATION_BATCH_SIZE': 1024,
    'USE_WANDB': True,
    'WANDB_PROJECT': 'ctr-dcn-monitoring',
    'WANDB_RUN_NAME': None,
    'WANDB_LOG_EVERY': 1,
    'WANDB_VIZ_EVERY': 5,
    'WANDB_THRESHOLD': 0.5,
    'USE_AMP': True,
    'AMP_DTYPE': 'float16',
    'EMA_ENABLED': True,
    'EMA_DECAY': 0.999,
    'CROSS_NUM_EXPERTS': 4,
    'CROSS_LOW_RANK': 64,
    'CROSS_GATING_HIDDEN': 128,
    'CROSS_DROPOUT': 0.1,
    'CROSS_LAYERS': 4,
    'MODEL_DROPOUT': 0.25,
    'EMBEDDING_DROPOUT': 0.05,
    'LSTM_HIDDEN_SIZE': 64,
    'DEEP_HIDDEN_DIMS': [768, 512, 256, 128],
    'LR_WARMUP_EPOCHS': 3,
    'LR_WARMUP_START_FACTOR': 0.1,
    'COSINE_T0': 10,
    'COSINE_T_MULT': 1,
    'COSINE_MIN_LR': 1e-6,
}


# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 시드 고정
seed_everything(CFG['SEED'])
