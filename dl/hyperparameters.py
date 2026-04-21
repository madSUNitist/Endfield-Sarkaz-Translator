PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# model & dataset paths
SRC_FILE = "corpus/datasets/train.skz"
TGT_FILE = "corpus/datasets/train.zh"
SP_SRC_MODEL = "corpus/datasets/sp_src.model"
SP_TGT_MODEL = "corpus/datasets/sp_zh.model"
SAVE_PATH = "models/model.pt"

# dataloader
# DATA_LOADER_BATCH_SIZE = 32
MAX_SEQ_LEN = 128
NUM_WORKERS = 4

# model
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
SP_MAX_LEN = 5000

# train
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
WARMUP_STEPS = 4000
BETAS = (0.9, 0.98)
EPS = 1e-9