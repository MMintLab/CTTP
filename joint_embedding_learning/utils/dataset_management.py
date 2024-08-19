from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
import re
import os
import yaml

def sort_order(filename):  #TODO: move this to utils
    return int(re.findall(r'\d+', filename)[-1])

def spatially_aligned_images(bubbles_img, gelslim_img):
    bubbles_size = bubbles_img.shape
    gelslim_size = gelslim_img.shape
    bubbles_resize = transforms.Resize((600,750))
    gelslim_resize = transforms.Resize((135,180))
    bubbles_resized = bubbles_resize(bubbles_img)
    gelslim_resized = gelslim_resize(gelslim_img)
    us_r = 78
    vs_r = 67
    us_l = 71
    vs_l = 68
    #center location on resized bubbles
    vs_r_resized = round(vs_r*(30/7))
    us_r_resized = round(us_r*(30/7))
    vs_l_resized = round(vs_l*(30/7))
    us_l_resized = round(us_l*(30/7))
    #center location on resized gelslim
    g_center_vs = 67
    g_center_us = 90
    #offset to fit alignment to most samples
    offset_v_r = 0
    offset_u_r = -20
    offset_v_l = +10
    offset_u_l = -36
    #get padded image
    gelslim_padded = torch.zeros((2,3,600,750), device=bubbles_img.device)
    top_r = vs_r_resized - g_center_vs + offset_v_r
    bottom_r = vs_r_resized + g_center_vs + offset_v_r
    left_r = us_r_resized - g_center_us + offset_u_r
    right_r = us_r_resized + g_center_us + offset_u_r
    top_l = vs_l_resized - g_center_vs + offset_v_l
    bottom_l = vs_l_resized + g_center_vs + offset_v_l
    left_l = us_l_resized - g_center_us + offset_u_l
    right_l = us_l_resized + g_center_us + offset_u_l
    gelslim_padded[0,:,top_r:bottom_r+1, left_r:right_r] = gelslim_resized[0]
    gelslim_padded[1,:,top_l:bottom_l+1, left_l:right_l] = gelslim_resized[1]
    # resize images back
    bubbles_resized = transforms.Resize(bubbles_size[-2:])(bubbles_resized)
    gelslim_padded = transforms.Resize(gelslim_size[-2:])(gelslim_padded)
    return bubbles_resized, gelslim_padded

def get_images_from_full_data(bubbles_data, gelslim_data, bubbles_transform = None, gelslim_transform = None, side = 'both', padding = False, difference = True):
    rotate = transforms.RandomRotation((180,180))
    bubbles_img = bubbles_data['bubble_imprint']

    gelslim_img = gelslim_data['gelslim']
    gelslim_ref = gelslim_data['gelslim_ref']
    gelslim_diff = gelslim_img - gelslim_ref

    if difference:
        gelslim_out = rotate(gelslim_diff / 255)
    else:
        gelslim_out = rotate(gelslim_img / 255)

    bubbles_angle = bubbles_data['theta']
    gelslim_angle = gelslim_data['theta']
    
    if bubbles_transform:
        bubbles_img = bubbles_transform(bubbles_img)

    if gelslim_transform:
        gelslim_out = gelslim_transform(gelslim_out)

    if padding:
        bubbles_img, gelslim_out = spatially_aligned_images(bubbles_img, gelslim_out)

    if side == 'right':
        bubbles_img = bubbles_img[0]
        gelslim_out = gelslim_out[0]
    elif side == 'left':
        bubbles_img = bubbles_img[1]
        gelslim_out = gelslim_out[1]
    
    return bubbles_img, gelslim_out, bubbles_angle, gelslim_angle

def get_datasets_stats(datasets, batch_size = 64):
    bubbles_img_totat_psum = 0
    bubbles_img_totat_psum_sq = 0
    bubbles_img_total_count = 0
    gelslim_diff_totat_psum = 0
    gelslim_diff_totat_psum_sq = 0
    gelslim_diff_total_count = 0
    bubbles_img_min = 1000
    bubbles_img_max = -1000
    gelslim_diff_min = [1000, 1000, 1000]
    gelslim_diff_max = [-1000, -1000, -1000]

    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i, ((bubbles_img, gelslim_diff), _) in enumerate(dataloader):
            bubbles_img_min = min(bubbles_img_min, bubbles_img.min())
            bubbles_img_max = max(bubbles_img_max, bubbles_img.max())
            bubbles_img_psum = bubbles_img.sum()
            bubbles_img_psum_sq = (bubbles_img**2).sum()
            bubbles_img_count = bubbles_img.numel()

            gelslim_diff_min = [min(gelslim_diff_min[0], gelslim_diff[:,0].min()), min(gelslim_diff_min[1], gelslim_diff[:,1].min()), min(gelslim_diff_min[2], gelslim_diff[:,2].min())]
            gelslim_diff_max = [max(gelslim_diff_max[0], gelslim_diff[:,0].max()), max(gelslim_diff_max[1], gelslim_diff[:,1].max()), max(gelslim_diff_max[2], gelslim_diff[:,2].max())]
            gelslim_diff_psum = gelslim_diff.sum([0,2,3])
            gelslim_diff_psum_sq = (gelslim_diff**2).sum([0,2,3])
            gelslim_diff_count = gelslim_diff[:,0].numel()
    
            bubbles_img_totat_psum += bubbles_img_psum
            bubbles_img_totat_psum_sq += bubbles_img_psum_sq
            bubbles_img_total_count += bubbles_img_count

            gelslim_diff_totat_psum += gelslim_diff_psum
            gelslim_diff_totat_psum_sq += gelslim_diff_psum_sq
            gelslim_diff_total_count += gelslim_diff_count

    bubbles_img_mean = bubbles_img_totat_psum / bubbles_img_total_count
    bubbles_img_std = torch.sqrt((bubbles_img_totat_psum_sq / bubbles_img_total_count) - bubbles_img_mean**2)

    gelslim_diff_mean = gelslim_diff_totat_psum / gelslim_diff_total_count
    gelslim_diff_std = torch.sqrt((gelslim_diff_totat_psum_sq / gelslim_diff_total_count) - gelslim_diff_mean**2)

    return bubbles_img_mean, bubbles_img_std,bubbles_img_min, bubbles_img_max, gelslim_diff_mean, gelslim_diff_std, gelslim_diff_min, gelslim_diff_max

def get_data_in_range(bubbles_stats, gelslims_stats, device = 'cuda:0'):
    bubbles_img_mean = torch.tensor(bubbles_stats['mean'], device=device)
    bubbles_img_std = torch.tensor(bubbles_stats['std'], device=device)
    bubbles_img_min = torch.tensor(bubbles_stats['min'], device=device)
    bubbles_img_max = torch.tensor(bubbles_stats['max'], device=device)

    gelslim_diff_mean = torch.tensor(gelslims_stats['mean'], device=device)
    gelslim_diff_std = torch.tensor(gelslims_stats['std'], device=device)
    gelslim_diff_min = torch.tensor(gelslims_stats['min'], device=device)
    gelslim_diff_max = torch.tensor(gelslims_stats['max'], device=device)

    bubbles_img_normed_min = (bubbles_img_min - bubbles_img_mean) / bubbles_img_std
    bubbles_img_normed_max = (bubbles_img_max - bubbles_img_mean) / bubbles_img_std

    gelslim_diff_normed_min = (gelslim_diff_min - gelslim_diff_mean) / gelslim_diff_std
    gelslim_diff_normed_max = (gelslim_diff_max - gelslim_diff_mean) / gelslim_diff_std

    gelslim_diff_normed_min = gelslim_diff_normed_min.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    gelslim_diff_normed_max = gelslim_diff_normed_max.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    return bubbles_img_normed_min, bubbles_img_normed_max, gelslim_diff_normed_min, gelslim_diff_normed_max

        
def get_transformations(dataset_name, device = 'cuda:0'):
    dataset_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'datasets_config.yaml')

    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)

    dataset_details = config[dataset_name]
    bubbles_transforms = dataset_details['bubbles_transforms']
    gelslim_transforms = dataset_details['gelslim_transforms']
    bubbles_stats = dataset_details['bubbles_stats']
    gelslim_stats = dataset_details['gelslim_stats']
    image_size = dataset_details['image_size']
    bubbles_img_normed_min, bubbles_img_normed_max, gelslim_diff_normed_min, gelslim_diff_normed_max = get_data_in_range(bubbles_stats, gelslim_stats, device = device)

    bubbles_transforms_list = []
    gelslim_transforms_list = []

    if bubbles_transforms is None:
        bubbles_transform = None
    else:
        if 'resize' in bubbles_transforms:
            bubbles_transforms_list.append(transforms.Resize((image_size, image_size)))
        
        if 'normalize' in bubbles_transforms:
            bubbles_transforms_list.append(transforms.Normalize(torch.tensor(bubbles_stats['mean']), torch.tensor(bubbles_stats['std'])))

        if 'in_range' in bubbles_transforms:
            bubbles_transforms_list.append(transforms.Lambda(lambda x: (x - bubbles_img_normed_min) / (bubbles_img_normed_max - bubbles_img_normed_min)))
        
        bubbles_transform = transforms.Compose(bubbles_transforms_list)
    
    if gelslim_transforms is None:
        gelslim_transform = None
    else:
        if 'resize' in gelslim_transforms:
            gelslim_transforms_list.append(transforms.Resize((image_size, image_size)))

        if 'normalize' in gelslim_transforms:
            gelslim_transforms_list.append(transforms.Normalize(torch.tensor(gelslim_stats['mean']), torch.tensor(gelslim_stats['std'])))
        
        if 'in_range' in gelslim_transforms:
            gelslim_transforms_list.append(transforms.Lambda(lambda x: (x - gelslim_diff_normed_min) / (gelslim_diff_normed_max - gelslim_diff_normed_min)))

        gelslim_transform = transforms.Compose(gelslim_transforms_list)

    return bubbles_transform, gelslim_transform