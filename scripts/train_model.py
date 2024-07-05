from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.utils.models_setup import get_trainer
import torch
import os
import yaml
import wandb

if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='CMTJE')
    parser.add_argument('--dataset_name', type=str, default='dataset_2', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='simclr', help='Name of the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--run_name', type=str, default='', help='Name of the run')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # Dataset configuration
    dataset_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'datasets_config.yaml')
    with open(dataset_config, 'r') as file:
        config = yaml.safe_load(file)
    dataset_details = config[args.dataset_name]


    # Model configuration
    model_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'models_config.yaml')
    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)
    model_details = config[args.model_name]

    if args.debug:
        project = 'CMTEJ_debug'
        output_folder = args.model_name + "_" + dataset_details['dataset_type'] + "_" + args.dataset_name + "_" + args.run_name
    else:
        project = 'CMTEJ_code_checking'
        output_folder = args.model_name + "_" + dataset_details['dataset_type'] + "_" + args.dataset_name + "_" + args.run_name

    wandb.init(
            project=project,
            name=output_folder,
            id=output_folder,
            resume=False,
        )

    # Dataset loading
    train_datasets, val_unseen_grasps, val_unseen_tools = get_all_contrastive_datasets(args.dataset_name, device=args.device)

    validation_datasets = {
        'val_unseen_grasps': val_unseen_grasps,
        'val_unseen_tools': val_unseen_tools
    }

    # Model Training
    trainer = get_trainer(args.model_name)
    trainer(train_datasets, validation_datasets, args.dataset_name, dataset_details, args.model_name, model_details, wandb, run_name=args.run_name, device=args.device)