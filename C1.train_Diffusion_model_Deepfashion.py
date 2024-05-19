import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
from diffusers.utils import check_min_version
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.diffusion.dataset import FPDM_Dataset, FPDM_Collate_fn
from src.diffusion.model import FPDM

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")


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
    # parser.add_argument("--config", type=str, default='./config/simclr.yaml', help='Path to config file')
    parser.add_argument("--project_name", type=str, default='deepfashion-diffusion-model-learning-dropmodule',
                        help='Path to config file')
    parser.add_argument("--root_path", type=str, default='./dataset/deepfashion/', help='Path to config file')
    parser.add_argument("--phase", type=str, default='train', help='train/test')
    parser.add_argument("--disable_logger", type=str2bool, default='false')
    parser.add_argument("--finetune_from", type=str, default='false')
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    return parser.parse_args()


def load_logger(args):
    # project_name = os.path.basename(args.config).split('.')[0]
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"logs/{args.project_name}/{time_now}/"
    os.makedirs(log_dir, exist_ok=True)
    if not args.disable_logger:
        logger = WandbLogger(name=args.project_name,
                             project=args.project_name,
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

args = get_parser()
args.batch_size = 16 #64
args.num_workers = 8
args.hidden_dim = 768
args.lr = 0 # 1e-4
args.temperature = 0.07
args.weight_decay = 1e-3
args.scheduler_t0 = 20
args.scheduler_t_mult = 2
args.scheduler_eta_max = 0.0001
args.scheduler_t_up = 5
args.scheduler_gamma = 0.5
args.max_epochs = 100
args.data_root_path = './dataset/deepfashion'
args.train_json_path = './dataset/deepfashion/train_pairs_data.json'
args.test_json_path = './dataset/deepfashion/test_pairs_data.json'
args.img_width = 256
args.img_height = 256
args.img_eval_size = (176, 256)  # (352, 512)
args.guidance_scale = 2.0
args.num_inference_steps = 50
args.seed_number = 7
args.num_images_per_prompt = 4
args.test_n_samples = 10
args.noise_offset = 0.1
args.imgs_drop_rate = 0.1
args.pose_drop_rate = 0.1
args.src_image_encoder_path = 'openai/clip-vit-base-patch16'
args.init_src_image_encoder = False
args.fusion_image_encoder = True
args.fusion_image_patch_encoder = True
args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
args.fusion_model_path = '../sign-diff/logs/deepfashion-fusion-CLIP-patch-learning-0516/2024-05-16T11-06-54/last.ckpt'
args.visualize_images = True
args.calculate_metrics = True
args.loss_type = 'mse_loss'  # mse_loss, shrinkage_loss
args.shrinkage_a = 50
args.shrinkage_c = 0.05
pl.seed_everything(args.seed_number)

logger, ckpt_cb = load_logger(args)

traindataset = FPDM_Dataset(
    args.train_json_path,
    args.data_root_path,
    phase='train',
    size=(args.img_width, args.img_height),  # w h
    imgs_drop_rate=args.imgs_drop_rate,
    pose_drop_rate=args.pose_drop_rate,
)

train_dataloader = DataLoader(traindataset,
                              collate_fn=FPDM_Collate_fn,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

testdataset = FPDM_Dataset(
    args.test_json_path,
    args.data_root_path,
    phase='test',
    size=(args.img_width, args.img_height),  # w h
    imgs_drop_rate=0,
    pose_drop_rate=0,
)

test_dataloader = DataLoader(
    testdataset,
    collate_fn=FPDM_Collate_fn,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=True)

lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=args.max_epochs,
    callbacks=[lr_monitor_cb, ckpt_cb],
    logger=logger, precision="16-mixed")  # precision="16-mixed"

# Setting the seed
# Check whether pretrained model exists. If yes, load it and skip training
model = FPDM(args)
trainer.fit(model, train_dataloader,
            test_dataloader)  # , ckpt_path='./logs/deepfashion-fusion-CLIP-patch-learning-clip16/2024-05-16T13-47-50/last.ckpt'
