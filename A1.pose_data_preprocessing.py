import glob
import os

import cv2
import tqdm

from src.fusion.datautil import ProcessingKeypoints

dataname = 'deepfashion'
save_dir = 'pose_img'
save_path = f'./dataset/{dataname}/{save_dir}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

kpt_txts = glob.glob(f'./dataset/{dataname}/pose/**/*.txt', recursive=True)

param = {}
PK = ProcessingKeypoints()
for i in tqdm.tqdm(range(len(kpt_txts))):
    pose_image_path = kpt_txts[i].replace('pose', save_dir).replace('.txt', '.jpg')
    img_path = kpt_txts[i].replace('/pose/', '/resized_img/').replace('txt', 'jpg')
    img = cv2.imread(img_path)
    h, w, c = img.shape
    pos_img = PK.get_label_tensor(kpt_txts[i], img, param)
    if not os.path.exists(os.path.dirname(pose_image_path)):
        os.makedirs(os.path.dirname(pose_image_path), exist_ok=True)
    pos_img.save(pose_image_path, format='JPEG', quality=100)
