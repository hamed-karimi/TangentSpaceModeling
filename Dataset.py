from collections import OrderedDict

from torch.utils import data
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import numpy as np



class ShapeNetMultiViewDataset(data.Dataset):
    def __init__(self, data_models_path_list, transform=None):
        self.data_models_path_list = data_models_path_list
        self.transform = transform
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.encoding_model = encoding_model #load_encoding_model(self.device)


    def __len__(self):
        return len(self.data_models_path_list)

    def __getitem__(self, idx):
        viewpoint_path1, viewpoint_path2 = self.data_models_path_list[idx]
        try:
            viewpoint1 = Image.open(viewpoint_path1).convert('RGB')
            viewpoint2 = Image.open(viewpoint_path2).convert('RGB')
        except:
            print(viewpoint_path1, 'or', viewpoint_path2, 'does not exist')
            return None, None
        if self.transform:
            viewpoint1 = self.transform(viewpoint1)
            viewpoint2 = self.transform(viewpoint2)
            # with torch.no_grad():
            #     viewpoint1 = self.encoding_model(self.transform(viewpoint1))
            #     viewpoint2 = self.encoding_model(self.transform(viewpoint2))

        return viewpoint_path1, viewpoint1, viewpoint_path2, viewpoint2


def load_dataset(split_name: str):
    assert split_name in ['train', 'val', 'test']
    split_dir = os.path.join('Dataset Splits', split_name)
    file_paths = np.load(os.path.join(split_dir, split_name + '.npy'), allow_pickle=True)
    split_transform = get_split_transforms()
    split_dataset = ShapeNetMultiViewDataset(file_paths.tolist(), transform=split_transform)
    return split_dataset

def save_dataset(split_name: str, path_list: list):
    path_np = np.array(path_list, dtype=object)
    split_dir = os.path.join('Dataset Splits', split_name)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    np.save(os.path.join(split_dir, split_name + '.npy'), path_np)

def get_split_transforms():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return transform

def generate_datasets(dataset_path, use_prev_indices=False, test=False):
    if use_prev_indices:
        datasets = {'train': None, 'val': None, 'test': None}
        for split_name in ['train', 'val']:
            datasets[split_name] = load_dataset(split_name=split_name)
    else:

        dataset_split_file_paths = {'train': [], 'val': [], 'test': []}
        datasets = {'train': None, 'val': None, 'test': None}
        print('generating datasets...')
        data_categories_path_list = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if '.' not in x]
        data_models_dir_list = np.array([os.path.join(x, y) for x in data_categories_path_list for y in
                                      os.listdir(x) if '.' not in y], dtype=object)
        # data_models_path_list = [] #np.empty_like(data_models_dir_list, dtype=object)
        for ind, split_name in enumerate(['train', 'val', 'test']):
            for i in range(data_models_dir_list.shape[0]):
                rotation_dir = os.path.join(str(data_models_dir_list[i]), 'models', '0')
                try:
                    all_image_names = sorted([name for name in os.listdir(rotation_dir) if name.endswith('.png')])
                    even_frames_paths = [(os.path.join(rotation_dir, all_image_names[j]),
                                         os.path.join(rotation_dir, all_image_names[j+1])) for j in
                                         range(0, len(all_image_names) - 1, 1)]

                    # odd_frames_paths = [(os.path.join(rotation_dir, all_image_names[j]),
                    #                       os.path.join(rotation_dir, all_image_names[j + 2])) for j in
                    #                      range(1, len(all_image_names) - 2, 2)]

                    dataset_split_file_paths[split_name].extend(even_frames_paths)
                    # dataset_split_file_paths[split_name].extend(odd_frames_paths)

                except:
                    print(rotation_dir, 'does not exist')
                    continue
                if test and i == 10:
                    break

        # encoding_model = load_encoding_model()
        for split_name in ['train', 'val', 'test']:
            split_transform = get_split_transforms()
            datasets[split_name] = ShapeNetMultiViewDataset(dataset_split_file_paths[split_name],
                                                            transform=split_transform)

            save_dataset(split_name, dataset_split_file_paths[split_name])
        print('train size: ', len(datasets['train']), 'val size: ', len(datasets['val']), 'test size: ', len(datasets['test']))
    return datasets