from torch.utils.data import Dataset, random_split, Subset
import torchvision
from joint_embedding_learning.utils.dataset_management import sort_order, get_images_from_full_data, get_transformations
from simclr.modules.transformations import TransformsSimCLR
import torch
import glob as glob
import os
import yaml

class CMTJESimCLR(Dataset):
    def __init__(self, root_dir_bubbles, root_dir_gelslim, selected_tools, bubbles_transforms=None, gelslim_transform=None, angle_start = 0, angle_skip = 1, side = 'both', device = 'cpu', padding = False, difference = True):
        self.side = side
        self.bubbles_files = []
        self.gelslim_files = []
        self.tool_per_sample = []
        self.idx_per_tool = []

        for i, tool in enumerate(selected_tools):
            len_tool_dataset = len(glob.glob(os.path.join(root_dir_bubbles, 'bubble_style_transfer_dataset_bubbles_' + tool, '*.pt')))
            self.tool_per_sample += [i] * len_tool_dataset
            self.idx_per_tool += list(range(len_tool_dataset))
            self.bubbles_files += sorted(glob.glob(os.path.join(root_dir_bubbles, 'bubble_style_transfer_dataset_bubbles_' + tool, '*.pt')), key=sort_order)
            self.gelslim_files += sorted(glob.glob(os.path.join(root_dir_gelslim, 'gelslim_style_transfer_dataset_gelslim_' + tool, '*.pt')), key=sort_order)
        
        files_idx = list(range(angle_start, len(self.bubbles_files), angle_skip))
        self.tool_per_sample = [self.tool_per_sample[i] for i in files_idx]
        self.bubbles_files = [self.bubbles_files[i] for i in files_idx]
        self.gelslim_files = [self.gelslim_files[i] for i in files_idx]
        self.bubblest_transforms = bubbles_transforms
        self.gelslim_transform = gelslim_transform
        self.device = device
        self.padding = padding
        self.difference = difference

    def __len__(self):        
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        bubbles_data = torch.load(self.bubbles_files[idx], map_location=self.device)
        gelslim_data = torch.load(self.gelslim_files[idx], map_location=self.device)

        # import pdb; pdb.set_trace()

        bubbles_img, gelslim_diff, _, _ = get_images_from_full_data(bubbles_data, gelslim_data, bubbles_transform = self.bubblest_transforms, gelslim_transform = self.gelslim_transform, side = self.side, padding = self.padding, difference = self.difference)
        bubbles_img = bubbles_img.repeat(3, 1, 1)
        return [bubbles_img, gelslim_diff], [torch.tensor(self.tool_per_sample[idx]).to(self.device), gelslim_data['theta'].to(self.device)*(180/torch.pi), gelslim_data['x'].to(self.device), gelslim_data['y'].to(self.device)] 
    
def get_tactile_datasets(dataset_name, dataset_details, data_split = 'train', device = 'cpu'):
    root_dir_bubbles = dataset_details['bubbles_path']
    root_dir_gelslim = dataset_details['gelslim_path']
    training_tools = dataset_details['tools']['training_tools']
    test_tools = dataset_details['tools']['test_tools']
    angle_skip = dataset_details['num_angles_skip']
    padding = dataset_details['padding']
    difference = dataset_details['difference']
    bubbles_transforms, gelslim_transform = get_transformations(dataset_name, device=device)
    
    datasets_right_train = []
    datasets_left_train = []
    datasets_right_unseen_grasps_val = []
    datasets_left_unseen_grasps_val = []
    datasets_right_unseen_grasps_test = []
    datasets_left_unseen_grasps_test = []
    datasets_right_unseen_tools_val = []
    datasets_left_unseen_tools_val = []
    datasets_right_unseen_tools_test = []
    datasets_left_unseen_tools_test = []

    for i in range(angle_skip):
        dataset_right = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, training_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='right', device=device, padding=padding, difference=difference)
        dataset_left = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, training_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='left', device=device, padding=padding, difference=difference)
        train_len = int(0.8 * len(dataset_right))
        unseen_len = len(dataset_right) - train_len
        dataset_right_train, dataset_right_unseen_grasps = random_split(dataset_right, [train_len, unseen_len], generator=torch.Generator().manual_seed(0))
        dataset_left_train, dataset_left_unseen_grasps = random_split(dataset_left, [train_len, unseen_len], generator=torch.Generator().manual_seed(0))

        datasets_right_train += [dataset_right_train]
        datasets_left_train += [dataset_left_train]

        val_len = int(0.5 * len(dataset_right_unseen_grasps))
        test_len = len(dataset_right_unseen_grasps) - val_len
        dataset_right_unseen_grasps_val, dataset_right_unseen_grasps_test = random_split(dataset_right_unseen_grasps, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        dataset_left_unseen_grasps_val, dataset_left_unseen_grasps_test = random_split(dataset_left_unseen_grasps, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        datasets_right_unseen_grasps_val += [dataset_right_unseen_grasps_val]
        datasets_left_unseen_grasps_val += [dataset_left_unseen_grasps_val]
        datasets_right_unseen_grasps_test += [dataset_right_unseen_grasps_test]
        datasets_left_unseen_grasps_test += [dataset_left_unseen_grasps_test]

        dataset_right_unseen_tools = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, test_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='right', device=device, padding=padding, difference=difference)
        dataset_left_unseen_tools = CMTJESimCLR(root_dir_bubbles, root_dir_gelslim, test_tools, bubbles_transforms=bubbles_transforms, gelslim_transform=gelslim_transform, angle_start = i, angle_skip=angle_skip, side='left', device=device, padding=padding, difference=difference)
        val_len = int(0.5 * len(dataset_right_unseen_tools))
        test_len = len(dataset_right_unseen_tools) - val_len
        dataset_right_unseen_tools_val, dataset_right_unseen_tools_test = random_split(dataset_right_unseen_tools, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        dataset_left_unseen_tools_val, dataset_left_unseen_tools_test = random_split(dataset_left_unseen_tools, [val_len, test_len], generator=torch.Generator().manual_seed(0))
        datasets_right_unseen_tools_val += [dataset_right_unseen_tools_val]
        datasets_left_unseen_tools_val += [dataset_left_unseen_tools_val]
        datasets_right_unseen_tools_test += [dataset_right_unseen_tools_test]
        datasets_left_unseen_tools_test += [dataset_left_unseen_tools_test]

    datasets_train = datasets_right_train + datasets_left_train
    datasets_unseen_grasps_val = datasets_right_unseen_grasps_val + datasets_left_unseen_grasps_val
    datasets_unseen_grasps_test = datasets_right_unseen_grasps_test + datasets_left_unseen_grasps_test
    datasets_unseen_tools_val = datasets_right_unseen_tools_val + datasets_left_unseen_tools_val
    datasets_unseen_tools_test = datasets_right_unseen_tools_test + datasets_left_unseen_tools_test

    if data_split == 'train':
        return datasets_train, datasets_unseen_grasps_val, datasets_unseen_tools_val
    else:
        return datasets_train, datasets_unseen_grasps_test, datasets_unseen_tools_test
    
def get_stl_10_datasets(dataset_details, data_split = 'train'):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'datasets')
    img_size = dataset_details['image_size']
    train_percent = dataset_details['train_percent']

    if data_split == 'train':
        train_dataset = torchvision.datasets.STL10(path, split='unlabeled', download=True, transform=TransformsSimCLR(size=img_size))
        if train_percent == 1.0:
            train_len = int(0.9 * len(train_dataset))
            val_len = int(0.5*(len(train_dataset) - train_len))
            test_len = len(train_dataset) - train_len - val_len
            _ , val_dataset, test_dataset = random_split(train_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(0))
        else:
            train_len = int(train_percent * len(train_dataset))
            val_len = int(0.5 * (len(train_dataset) - train_len))
            test_len = len(train_dataset) - train_len - val_len
            train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(0))
    elif data_split == 'test':
        train_dataset = torchvision.datasets.STL10(
                                                        path,
                                                        split="train",
                                                        download=True,
                                                        transform=TransformsSimCLR(size=img_size).test_transform,
                                                    )
        test_dataset = torchvision.datasets.STL10(
                                                            path,
                                                            split="test",
                                                            download=True,
                                                            transform=TransformsSimCLR(size=img_size).test_transform,
                                                    )
        val_dataset = []
    return [train_dataset], [val_dataset], [test_dataset]
    
def get_all_contrastive_datasets(dataset_name, data_split = 'train', device = 'cpu'):
    dataset_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'datasets_config.yaml')

    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)

    dataset_details = config[dataset_name]
    dataset_type = dataset_details['dataset_type']

    if dataset_type == 'tactile':
        return get_tactile_datasets(dataset_name, dataset_details, data_split, device)
    elif dataset_type == 'stl_10':
        return get_stl_10_datasets(dataset_details, data_split)
    else:
        raise ValueError('Dataset type not recognized')

