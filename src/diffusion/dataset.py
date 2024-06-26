import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor


def FPDM_Collate_fn(data):
    processed_source_image = torch.stack([example["processed_s_img"] for example in data])
    # processed_source_image = processed_source_image.to(memory_format=torch.contiguous_format).float()

    # processed_target_image = torch.stack([example["processed_t_img"] for example in data])
    # processed_target_image = processed_target_image.to(memory_format=torch.contiguous_format).float()

    processed_target_pose = torch.stack([example["processed_t_pose"] for example in data])
    # processed_pose_image = processed_pose_image.to(memory_format=torch.contiguous_format).float()

    source_image = torch.stack([example["trans_s_img"] for example in data])
    # source_image = source_image.to(memory_format=torch.contiguous_format).float()

    target_pose = torch.stack([example["trans_t_pose"] for example in data])
    # target_pose = target_pose.to(memory_format=torch.contiguous_format).float()

    target_image = torch.stack([example["trans_t_img"] for example in data])
    # target_image = target_image.to(memory_format=torch.contiguous_format).float()

    return {
        "processed_source_image": processed_source_image,
        # "processed_target_image": processed_target_image,
        "processed_target_pose": processed_target_pose,
        "source_image": source_image,
        "target_pose": target_pose,
        "target_image": target_image,
    }


class FPDM_Dataset(Dataset):
    def __init__(
            self,
            json_file,
            image_root_path,
            phase='train',
            size=(512, 512),
            imgs_drop_rate=0.0,
            pose_drop_rate=0.0,
    ):
        if isinstance(json_file, str):
            self.data = json.load(open(json_file))
            print(len(self.data))
        else:
            self.data = json_file
        self.image_root_path = image_root_path

        self.phase = phase
        self.size = size

        self.imgs_drop_rate = imgs_drop_rate
        self.pose_drop_rate = pose_drop_rate

        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')  # 앞으로 빼기

        self.ColorJitter_functions = {0: transforms.functional.adjust_brightness,
                                      1: transforms.functional.adjust_contrast,
                                      2: transforms.functional.adjust_saturation,
                                      3: transforms.functional.adjust_hue}

        self.transform_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), ]
        )

    def transforms(self, source_img, target_img, pos_t_img):
        # Random crop
        crop = transforms.RandomResizedCrop(self.size)
        params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        source_img = transforms.functional.crop(source_img, *params)
        source_img = transforms.functional.resize(source_img, crop.size)

        params = crop.get_params(source_img, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        target_img = transforms.functional.crop(target_img, *params)
        target_img = transforms.functional.resize(target_img, crop.size)

        pos_t_img = transforms.functional.crop(pos_t_img, *params)
        pos_t_img = transforms.functional.resize(pos_t_img, crop.size)

        # Random horizontal flipping
        if random.random() < 0.5:
            source_img = transforms.functional.hflip(source_img)
        if random.random() < 0.5:
            target_img = transforms.functional.hflip(target_img)
            pos_t_img = transforms.functional.hflip(
                pos_t_img)  # plt.imshow(pos_t_img.numpy().transpose(1,2,0)); plt.show()

        if random.random() < 0.2:  # 0.8
            jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            params = jitter.get_params(jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)

            for i in np.array(params[0]):
                source_img = self.ColorJitter_functions[i](source_img, params[i + 1])
                target_img = self.ColorJitter_functions[i](target_img, params[i + 1])

        if random.random() < 0.1:  # 0.2
            target_img = transforms.functional.rgb_to_grayscale(target_img, num_output_channels=3)
            source_img = transforms.functional.rgb_to_grayscale(source_img, num_output_channels=3)

        # target_img = transforms.functional.to_tensor(target_img)
        # source_img = transforms.functional.to_tensor(source_img)
        # pos_t_img = transforms.functional.to_tensor(pos_t_img)
        #
        # target_img = transforms.functional.normalize(target_img, (0.5,), (0.5,))
        # source_img = transforms.functional.normalize(source_img, (0.5,), (0.5,))
        # pos_t_img = transforms.functional.normalize(pos_t_img, (0.5,), (0.5,))

        return source_img, target_img, pos_t_img

    def __getitem__(self, idx):
        item = self.data[idx]

        s_img_path = os.path.join(self.image_root_path, item["source_image"])  # .replace(".jpg", ".jpg")) # png
        s_img = Image.open(s_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        t_img_path = os.path.join(self.image_root_path, item["target_image"])
        t_img = Image.open(t_img_path).convert("RGB").resize(self.size, Image.BICUBIC)

        t_pose = Image.open(t_img_path.replace("/img/", "/pose_img/")).convert("RGB").resize(self.size, Image.BICUBIC)
        # if self.args.phase == 'train':
        if self.phase == 'train':
            s_img, t_img, t_pose = self.transforms(s_img, t_img, t_pose)

        trans_s_img = self.transform_normalize(s_img)
        trans_t_img = self.transform_normalize(t_img)
        trans_t_pose = self.transform_normalize(t_pose)

        processed_s_img = (self.image_processor(images=s_img,
                                                return_tensors="pt").pixel_values).squeeze(dim=0)
        processed_t_pose = (self.image_processor(images=t_pose,
                                                 return_tensors="pt").pixel_values).squeeze(dim=0)
        ## dropout s_img for clip
        if random.random() < self.imgs_drop_rate:
            processed_s_img = torch.zeros(processed_s_img.shape)
        ## dropout pos_img for clip
        if random.random() < self.pose_drop_rate:
            processed_t_pose = torch.zeros(processed_t_pose.shape)

        return {
            "processed_s_img": processed_s_img,
            # "processed_t_img": processed_t_img,
            "processed_t_pose": processed_t_pose,
            "trans_t_img": trans_t_img,
            "trans_t_pose": trans_t_pose,
            "trans_s_img": trans_s_img,
        }

    def __len__(self):
        return len(self.data)
