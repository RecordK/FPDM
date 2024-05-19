import glob
import json
import os

import cv2
import numpy as np
import tqdm



# 데이터 train, test 명확하게 하기


def reformat_dataset(dataset_dir, image_list):
    image_dict = {}
    for i in range(len(image_list)):
        _path0 = image_list[i].replace(dataset_dir, '.')
        _path_key = image_list[i].replace('./dataset/deepfashion/img', '').replace('_', '').replace('/', '')
        image_dict[_path_key] = _path0

    filenames_train = []
    file_txt = '{}/annotations/fasion-resize-pairs-train.csv'.format(dataset_dir)
    data = np.loadtxt(file_txt, dtype=str, skiprows=1)
    for i in data:
        source = i.split(',')[0]
        target = i.split(',')[1]
        source = source.replace('fashion', '').replace('_', '')
        target = target.replace('fashion', '').replace('_', '')
        source_path = image_dict[source]
        target_path = image_dict[target]
        filenames_train.append({'source_image': source_path,
                                'target_image': target_path})

    filenames_test = []
    file_txt = '{}/annotations/fasion-resize-pairs-test.csv'.format(dataset_dir)
    data = np.loadtxt(file_txt, dtype=str, skiprows=1)
    for i in data:
        source = i.split(',')[0]
        target = i.split(',')[1]
        source = source.replace('fashion', '').replace('_', '')
        target = target.replace('fashion', '').replace('_', '')
        source_path = image_dict[source]
        target_path = image_dict[target]
        filenames_test.append({'source_image': source_path,
                               'target_image': target_path})
    return filenames_train, filenames_test

def omit_image_only(_dataset, type):
    intr_path = []
    idx_bool = []
    for i in range(len(_dataset)):
        flag = True
        for j in ['source_image', 'target_image']:
            _path = _dataset[i][j]
            _path = _path.replace('img','pose_img')
            if not os.path.exists(os.path.join(dataset_dir, _path)):
                # print(f'warning : no pose matched with img : {_path}')
                intr_path.append(_path)
                flag = False
        if flag:
            idx_bool.append(True)
        else:
            idx_bool.append(False)
    print(f'omit {type}, {len(set(intr_path))} images, {len(idx_bool)-sum(idx_bool)}/{len(idx_bool)} pairs')
    _dataset = list(np.array(_dataset)[idx_bool])
    return _dataset



def resize_image(image_list, resized_dirname, ratio):
    for i in tqdm.tqdm(range(len(image_list))):
        c_path = image_list[i]
        s_path = c_path.replace('img', resized_dirname)
        if os.path.exists(s_path):
            continue
        img = cv2.imread(c_path)
        h, w, c = img.shape
        img = cv2.resize(img, (int(w*ratio), int(h*ratio)))

        dir_path = os.path.split(s_path)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(s_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if i % 100 == 0:
            print(i)
    print('finishing the image resize process')

root_dir = './dataset'
dataset = 'deepfashion'
resized_dirname = 'resized_img'
dataset_dir = os.path.join(root_dir, dataset)

print('image resize preprocessing')
image_list = glob.glob(os.path.join(dataset_dir,'img/**/*.jpg'), recursive=True)
resize_image(image_list, resized_dirname, ratio=0.5)

print('annotation preprocessing')
train_dataset, test_dataset = reformat_dataset(dataset_dir, image_list)
# check if imgs has no pose annotation omit.
train_dataset = omit_image_only(train_dataset, 'train')
test_dataset = omit_image_only(test_dataset, 'test')

with open(os.path.join(dataset_dir, f'train_pairs_data.json'), 'w') as f:
    json.dump(train_dataset, f)
with open(os.path.join(dataset_dir, f'test_pairs_data.json'), 'w') as f:
    json.dump(test_dataset, f)

# _dataset = test_dataset
# for i in range(len(_dataset)):
#     dat = _dataset[0]
#     if not os.path.split(dat['source_image'])[0] == os.path.split(dat['target_image'])[0]:
#         if not os.path.split(dat['source_image'])[1][:2] == os.path.split(dat['target_image'])[1][:2]:
#             print('warnnings')