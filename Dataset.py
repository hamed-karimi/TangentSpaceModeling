import pickle
from collections import OrderedDict

from torch.utils import data
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import numpy as np



class ShapeNetMultiViewDataset(data.Dataset):
    def __init__(self, data_models_path_indices, transform=None, data_models_dir_list=None, rotation_sample_num=50):
        self.data_models_path_indices = data_models_path_indices
        self.data_models_dir_list = data_models_dir_list
        self.rotation_sample_num = rotation_sample_num
        self.transform = transform
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.encoding_model = encoding_model #load_encoding_model(self.device)


    def __len__(self):
        return len(self.data_models_path_indices)

    def __getitem__(self, idx):
        object_id, rotation_id, viewpoint_id = self.data_models_path_indices[idx]
        viewpoint_path1 = os.path.join(self.data_models_dir_list[object_id], 'models', str(rotation_id), str(viewpoint_id)+'.png')
        viewpoint_path2 = os.path.join(self.data_models_dir_list[object_id], 'models', str(rotation_id), str(viewpoint_id+1)+'.png')
        try:
            viewpoint1 = Image.open(viewpoint_path1).convert('RGB')
            viewpoint2 = Image.open(viewpoint_path2).convert('RGB')
        except:
            print(viewpoint_path1, 'or', viewpoint_path2, 'does not exist')
            return None, None
        if self.transform:
            viewpoint1 = self.transform(viewpoint1)
            viewpoint2 = self.transform(viewpoint2)


        return viewpoint_path1, viewpoint1, viewpoint_path2, viewpoint2


def load_dataset(split_name: str):
    assert split_name in ['train', 'val', 'test']
    split_dir = os.path.join('Dataset Splits', split_name)
    split_info = pickle.load(open(os.path.join(split_dir, 'split_info.pkl'), 'rb'))
    data_models_dir_list = np.load(os.path.join(split_dir, 'data_models_dir_list.npy'), allow_pickle=True)
    path_indices_list = np.load(os.path.join(split_dir, 'file_indices.npy'), allow_pickle=True)
    split_transform = get_split_transforms()
    split_dataset = ShapeNetMultiViewDataset(path_indices_list.tolist(),
                                             transform=split_transform,
                                             data_models_dir_list=data_models_dir_list,
                                             rotation_sample_num=split_info['rotation_sample_num'] )
    return split_dataset

def save_dataset(split_name: str, path_indices_list: list, data_models_dir_list, rotation_sample_num):
    split_info = {
        'data_models_dir_list_path': './data_models_dir_list.npy',
        'rotation_sample_num': rotation_sample_num,
    }
    path_indices_np = np.array(path_indices_list, dtype=int)
    split_dir = os.path.join('Dataset Splits', split_name)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    np.save(os.path.join(split_dir, 'file_indices.npy'), path_indices_np)
    np.save(os.path.join(split_dir, 'data_models_dir_list.npy'), data_models_dir_list)
    with open(os.path.join(split_dir, 'split_info.pkl'), 'wb') as f:
        pickle.dump(split_info, f)

def get_split_transforms():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return transform

def generate_datasets(dataset_path, rotation_sample_num=50, use_prev_indices=False, test=False):
    if use_prev_indices:
        datasets = {'train': None, 'val': None, 'test': None}
        for split_name in ['train', 'val']:
            datasets[split_name] = load_dataset(split_name=split_name)
    else:

        dataset_split_file_path_indices = {'train': [], 'val': [], 'test': []}
        datasets = {'train': None, 'val': None, 'test': None}
        print('generating datasets...')
        data_categories_path_list = sorted(os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if '.' not in x)
        data_models_dir_list = np.array([os.path.join(x, y) for x in data_categories_path_list for y in
                                      os.listdir(x) if '.' not in y], dtype=object)
        data_models_dir_list.sort()
        # data_models_path_list = [] #np.empty_like(data_models_dir_list, dtype=object)
        for ind, split_name in enumerate(['train', 'val', 'test']):
            for i in range(data_models_dir_list.shape[0]):
                rotation_models_dir = os.path.join(str(data_models_dir_list[i]), 'models')
                for j in range(rotation_sample_num):
                    rotation_dir = os.path.join(rotation_models_dir, str(j))
                    try:
                        image_count = len([f.name for f in os.scandir(rotation_dir) if f.name.endswith('.png')])
                        frame_path_indices = [[i, j, k] for k in range(image_count-1)]

                        dataset_split_file_path_indices[split_name].extend(frame_path_indices)

                    except:
                        print(rotation_dir, 'does not exist')
                        continue
                if test and i == 10:
                    break

        # encoding_model = load_encoding_model()
        for split_name in ['train', 'val', 'test']:
            split_transform = get_split_transforms()
            datasets[split_name] = ShapeNetMultiViewDataset(dataset_split_file_path_indices[split_name],
                                                            transform=split_transform,
                                                            data_models_dir_list=data_models_dir_list,
                                                            rotation_sample_num=rotation_sample_num)

            save_dataset(split_name,
                         dataset_split_file_path_indices[split_name],
                         data_models_dir_list,
                         rotation_sample_num)
        print('train size: ', len(datasets['train']), 'val size: ', len(datasets['val']), 'test size: ', len(datasets['test']))
    return datasets