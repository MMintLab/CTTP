from joint_embedding_learning.utils.dataset_management import sort_order, get_images_from_full_data
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import os

def vis_data(bubble_tool_path, gelslim_tool_path, vis_range = (0,10,1), vis_colm = 5, side = 'right', title = 'distinct_data_visualization', print_angles = False):
    bubbles_files = sorted(glob.glob(os.path.join(bubble_tool_path, '*.pt')), key=sort_order)
    gelslim_files = sorted(glob.glob(os.path.join(gelslim_tool_path, '*.pt')), key=sort_order)
    bubbles_imgs = []
    gelslim_diffs = []
    bubbles_angles = []
    gelslim_angles = []

    if vis_range is None:
        vis_range = (0, len(bubbles_files), 10)
        
    for idx in range(vis_range[0], vis_range[1], vis_range[2]):
        bubbles_data = torch.load(bubbles_files[idx])
        gelslim_data = torch.load(gelslim_files[idx])
        bubbles_img, gelslim_diff, bubbles_angle, gelslim_angle = get_images_from_full_data(bubbles_data, gelslim_data, bubbles_transform = None, gelslim_transform = None, side = side)
        bubbles_imgs.append(bubbles_img)
        gelslim_diffs.append(gelslim_diff)
        bubbles_angles.append(bubbles_angle.item())
        gelslim_angles.append(gelslim_angle.item())

    bubbles_imgs = torch.stack(bubbles_imgs)
    gelslim_diffs = torch.stack(gelslim_diffs)

    bubbles_imgs_grid = make_grid(bubbles_imgs, nrow=vis_colm, normalize=True)
    gelslim_diffs_grid = make_grid(gelslim_diffs, nrow=vis_colm, normalize=True)

    if print_angles:
        print('Angles_' + title + ':')
        for angle in gelslim_angles:
            print(f"{(angle * 180 / np.pi):.2f}")

    plt.imshow(bubbles_imgs_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(title)
    plt.savefig('./distinct_data_vis/bubbles_imgs_' + title + '.png')

    plt.imshow(gelslim_diffs_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(title)
    plt.savefig('./distinct_data_vis/gelslim_diffs_' + title + '.png')

    return

if __name__ == '__main__':
    #bubble_tool_path = '/home/samanta/tactile_style_transfer/new_processed_data/bubbles/data/bubble_style_transfer_dataset_bubbles_pattern_36' #pattern_01_2_lines_angle_1'
    #gelslim_tool_path = '/home/samanta/tactile_style_transfer/new_processed_data/gelslims/data/gelslim_style_transfer_dataset_gelslim_pattern_36' #pattern_01_2_lines_angle_1'

    bubble_tool_path = '/home/samanta/tactile_style_transfer/processed_data/bubbles/data/bubble_style_transfer_dataset_bubbles_pattern_01_2_lines_angle_1'
    gelslim_tool_path = '/home/samanta/tactile_style_transfer/processed_data/gelslims/data/gelslim_style_transfer_dataset_gelslim_pattern_01_2_lines_angle_1'

    #Checking angles diff
    vis_data(bubble_tool_path, gelslim_tool_path, vis_range = (0,10,1), vis_colm = 1, side = 'right', title='angles_diff_right_old', print_angles = True)
    vis_data(bubble_tool_path, gelslim_tool_path, vis_range = (0,10,1), vis_colm = 1, side = 'left', title='angles_diff_left_old', print_angles = True)
    #Checking spatial diff
    vis_data(bubble_tool_path, gelslim_tool_path, vis_range = None, vis_colm = 8, side = 'right', title='spatial_diff_right_old')
    vis_data(bubble_tool_path, gelslim_tool_path, vis_range = None, vis_colm = 8, side = 'left', title='spatial_diff_left_old')
