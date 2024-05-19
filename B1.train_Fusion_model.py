import argparse
import json
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.fusion.dataset import FusionDataset
from src.fusion.models import FusionModel


def read_dataset(root_path, filename):
    with open(os.path.join(root_path, filename), 'r') as f:
        data = json.load(f)
    return data


def load_logger(args):
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"logs/{args.project_name}/{time_now}/"
    os.makedirs(log_dir, exist_ok=True)
    if not args.disable_logger:
        if not args.trained_model_name:
            logger = WandbLogger(name=args.project_name,
                                 project=args.project_name,
                                 log_model=True,
                                 save_dir=log_dir)
        else:
            logger = WandbLogger(name=args.project_name,
                                 project=args.project_name,
                                 id=args.wandb_id,
                                 resume='allow',
                                 log_model=True,
                                 save_dir=log_dir)
    else:
        logger = None

    ckpt_cb = ModelCheckpoint(dirpath=log_dir,
                              monitor='val_loss',
                              mode="min",
                              save_top_k=1,
                              save_last=True)

    return logger, ckpt_cb


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='deepfashion-fusion-CLIP-patch-learning-sch',
                        help='Path to config file')
    parser.add_argument("--root_path", type=str, default='./dataset/deepfashion/', help='Path to config file')
    parser.add_argument("--phase", type=str, default='train', help='train/test')
    parser.add_argument("--disable_logger", type=str2bool, default='false')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    return parser.parse_args()


args = get_parser()
args.batch_size = 96
args.num_workers = 8
args.combiner_hidden_dim = 512  # large 768 base 512
args.lr = 0
args.scheduler_t0 = 10
args.scheduler_t_mult = 2
args.scheduler_eta_max = 0.0001
args.scheduler_t_up = 3
args.scheduler_gamma = 0.5
args.temperature = 0.07  # 0.07
args.weight_decay = 1e-2
args.max_epochs = 60
args.scale_size = (256, 256)
args.lambda_l1 = 0.0001
args.encoder_type = 'clip'
args.attn_hidden_dim = 768  # large-1024 base 768
args.mh_attn_size = 16
args.img_encoder_update = True
args.trained_model_name = None  #
args.wandb_id = None  # '.....'
args.train_dataset_name = 'train_pairs_data.json'
args.test_dataset_name = 'test_pairs_data.json'
args.img_encoder_path = 'openai/clip-vit-base-patch16'
args.train_patch_embeddings = True
args.train_patch_embeddings_sampling_ratio = 0.01

# if args.trained_model_name:
#     run_id = args.trained_model_name.split('-')[-1]
#     wandb.init(id=run_id, resume="allow")

logger, ckpt_cb = load_logger(args)
train_dataset = read_dataset(args.root_path, args.train_dataset_name)
test_dataset = read_dataset(args.root_path, args.test_dataset_name)
len(train_dataset)

dat = []
for i in train_dataset:
    dat.append(os.path.split(i['source_image'])[0])
    dat.append(os.path.split(i['target_image'])[0])
dat = list(set(dat))

train_dataset = FusionDataset(train_dataset, args)
train_dataloader = DataLoader(train_dataset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

test_dataset = FusionDataset(test_dataset, args)
test_dataloader = DataLoader(test_dataset,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)

lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=args.max_epochs,
    callbacks=[lr_monitor_cb, ckpt_cb],
    logger=logger,
)

# Setting the seed
pl.seed_everything(7)
if args.trained_model_name:
    ckpt_path = f'./logs/{args.project_name}/{args.trained_model_name}/last.ckpt'
    model = FusionModel(args)
    trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
else:
    model = FusionModel(args)
    trainer.fit(model, train_dataloader, test_dataloader)
