import glob
import json
import os

import cv2
import numpy as np
import tqdm

'''
데이터셋 전처리 코드
similarity rank :  id, id_angle, id_style, same_categoy, different_category
'''


# 데이터 train, test 명확하게 하기

def load_data(root, dataset):
    if dataset == 'deepfashion':
        file_txt = '{}/train_pairs.txt'.format(root)
        filenames_train = []
        with open(file_txt, 'r') as f:
            lines = f.readlines()
            for item in lines:
                filenames_train.extend(item.strip().split(','))
        filenames_train = list(set(filenames_train))
        filenames_test = []
        file_txt = '{}/test_pairs.txt'.format(root)
        with open(file_txt, 'r') as f:
            lines = f.readlines()
            for item in lines:
                filenames_test.extend(item.strip().split(','))
        filenames_test = list(set(filenames_test))
    return filenames_train, filenames_test


def reformat_data(root, dataset):
    if dataset == 'deepfashion':
        file_txt = '{}/train_pairs.txt'.format(root)
        filenames_train = []
        with open(file_txt, 'r') as f:
            lines = f.readlines()
            for item in lines:
                _st = item.strip().split(',')
                filenames_train.append({'source_image': _st[0],
                                        'target_image': _st[1]})

        filenames_test = []
        file_txt = '{}/test_pairs.txt'.format(root)
        with open(file_txt, 'r') as f:
            lines = f.readlines()
            for item in lines:
                _st = item.strip().split(',')
                filenames_test.append({'source_image': _st[0],
                                       'target_image': _st[1]})
    return filenames_train, filenames_test


def sample_item(_filename, val, dict_id, dict_id_same_style, dict_clothes):
    _, sex, clothes, id, img = _filename.split('/')
    _no, _, _ = img.split('_')
    id_no = id + '_' + _no

    # same style in id
    if val == 'same-style-with-id':
        _file_list0 = np.array(dict_id_same_style[id_no])
        mask = _file_list0 != _filename
        sampled_data = np.random.choice(_file_list0[mask])
    if val == 'differ-style-with-id':
        # differ style with id
        _file_list0 = np.array(dict_id[id])
        _file_list1 = np.array(dict_id_same_style[id_no])
        mask = ~np.isin(_file_list0, _file_list1)
        sampled_data = np.random.choice(_file_list0[mask])
    if val == 'differ-id':
        # differ id
        _file_list0 = np.array(dict_clothes[clothes])
        _file_list1 = np.array(dict_id[id])
        mask = ~np.isin(_file_list0, _file_list1)
        sampled_data = np.random.choice(_file_list0[mask])
    if val == 'differ-category':
        # differ-category
        _cate_list0 = np.array(list(dict_clothes.keys()))
        mask = _cate_list0 != clothes
        sampled_cate = np.random.choice(_cate_list0[mask])
        sampled_data = np.random.choice(dict_clothes[sampled_cate])
    return sampled_data
    # 데이터 리스트 만들기


def make_dictionary(_filename):
    _dict_filename = {}
    _dict_clothes = {}
    _dict_id = {}
    _dict_id_same_style = {}
    # _dict_id_different_style = {}
    _dict_image = {}

    for idx, _file in enumerate(_filename):
        _, sex, clothes, id, img = _file.split('/')
        no, angle, angle2 = img.split('_')
        id_no = id + '_' + no
        if clothes not in list(_dict_clothes.keys()):
            _dict_clothes[clothes] = []
        if img not in list(_dict_image.keys()):
            _dict_image[img] = []
        if _file not in list(_dict_filename.keys()):
            _dict_filename[_file] = []
        if id not in list(_dict_id.keys()):
            _dict_id[id] = []
        if id_no not in list(_dict_id_same_style.keys()):
            _dict_id_same_style[id_no] = []

        _dict_clothes[clothes].append(_file)
        _dict_id[id].append(_file)
        _dict_image[img].append(_file)
        _dict_filename[_file].append(_file)
        _dict_id_same_style[id_no].append(_file)

    return _dict_clothes, _dict_id, _dict_id_same_style, _dict_image, _dict_filename


img_list = glob.glob('./dataset/deepfashion/img/**/*.jpg', recursive=True)
for i in tqdm.tqdm(range(len(img_list))):
    c_path = img_list[i]
    img = cv2.imread(c_path)
    h, w, c = img.shape
    img = cv2.resize(img, (w // 2, h // 2))
    s_path = c_path.replace('/img', '/img_resize')
    dir_path = os.path.split(s_path)[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    cv2.imwrite(s_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    if i % 100 == 0:
        print(i)

glob.glob('./dataset/deepfashion/*')
root = './dataset/deepfashion/'
dataset = 'deepfashion'
filenames_train, filenames_test = load_data(root, dataset)
train_dataset, test_dataset = reformat_data(root, dataset)
with open(os.path.join(root, f'train_pairs_data.json'), 'w') as f:
    json.dump(train_dataset, f)
with open(os.path.join(root, f'test_pairs_data.json'), 'w') as f:
    json.dump(test_dataset, f)

phase = 'test'
sample_size_by_item = 6

if phase == 'train':
    filenames = filenames_train
else:
    filenames = filenames_test

results = {}
for idx, _file in enumerate(filenames):
    results[idx] = _file

dict_clothes, dict_id, dict_id_same_style, dict_image, dict_filename = make_dictionary(filenames)
idx = np.arange(len(filenames))
# np.random.shuffle(idx)
sample_val_dict = {0: 'same-style-with-id', 1: 'differ-style-with-id',
                   2: 'differ-id', 3: 'differ-category'}

# sampling
# results = np.zeros((total_sample_size,3)).astype(object)
results = []
step = 0
for i in idx:
    _filename = filenames[i]
    # for k in range(sample_size_by_item):
    _, sex, clothes, id, img = _filename.split('/')
    _no, _, _ = img.split('_')
    id_no = id + '_' + _no
    _file_list0 = np.array(dict_id_same_style[id_no])
    mask = _file_list0  # != _filename
    _file_list0 = _file_list0  # [mask]
    for target_file in _file_list0:
        results.append({'source_image': _filename,
                        'target_image': target_file})
        step += 1
        if step % 100 == 0:
            print(f'{step}')

with open(os.path.join(root, f'{phase}_all_pairs_data.json'), 'w') as f:
    json.dump(results, f)

# 데이터 샘플링- 학습 데이터셋 만들기
