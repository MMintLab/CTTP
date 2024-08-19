from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.utils.dataset_management import get_datasets_stats
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dataset_name = 'dataset_7'
    device = 'cuda:0'
    print('Dataset:', dataset_name)
    print('Device:', device)

    datasets_train, datasets_unseen_grasps, datasets_unseen_tools = get_all_contrastive_datasets(dataset_name, device=device)
    
    dataloader_right_train = DataLoader(datasets_train[0], batch_size=16, shuffle=False)
    [bubbles_img, gelslim_diff], label = next(iter(dataloader_right_train))
    # import pdb; pdb.set_trace()
    bubbles_img_grid = make_grid(bubbles_img, nrow=4, normalize=True)
    gelslim_diff_grid = make_grid(gelslim_diff, nrow=4, normalize=True)
    # print(label)

    bubbles_img_mean, bubbles_img_std,bubbles_img_min, bubbles_img_max, gelslim_diff_mean, gelslim_diff_std, gelslim_diff_min, gelslim_diff_max = get_datasets_stats(datasets_train)
    print('bubble_img_mean:', bubbles_img_mean)
    print('bubble_img_std:', bubbles_img_std)
    print('bubble_img_min:', bubbles_img_min)
    print('bubble_img_max:', bubbles_img_max)
    print('gelslim_diff_mean:', gelslim_diff_mean)
    print('gelslim_diff_std:', gelslim_diff_std)
    print('gelslim_diff_min:', gelslim_diff_min)
    print('gelslim_diff_max:', gelslim_diff_max)

    plt.imshow(bubbles_img_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('bubbles_img')
    plt.tight_layout()
    plt.savefig('./dataset_check/bubbles_img.png')

    plt.imshow(gelslim_diff_grid.permute(1, 2, 0).cpu().numpy(), alpha=0.8)
    plt.axis('off')
    plt.title('gelslim_diff')
    plt.tight_layout()
    plt.savefig('./dataset_check/gelslim_diff.png')

    print(bubbles_img.shape)
    print(gelslim_diff.shape)