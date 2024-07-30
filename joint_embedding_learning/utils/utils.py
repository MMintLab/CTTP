import os
import torch
from simclr import SimCLR
from simclr.modules import LARS as LARS_simclr
import yaml
from simclr.modules import NT_Xent
from joint_embedding_learning.barlow_twins.barlow import BarlowLoss, LARS


def load_optimizer(args, model):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # TODO: LARS
        scheduler = None
        scaler = None
    elif args.optimizer == "LARS":
        scheduler = None
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)
        
        scaler = torch.cuda.amp.GradScaler()
    else:
        raise NotImplementedError
    
    return optimizer, scheduler, scaler

def get_loss(args, model_name):
    if model_name == 'simclr':
        criterion = NT_Xent(args.batch_size, args.temperature, 1)
    elif model_name == 'barlow_twins':
        criterion = BarlowLoss(args.projector, args.batch_size, args.lambd, args.device)
    else:
        raise ValueError(f"Model {model_name} not found")
    return criterion

def save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=""):
    path_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_type, dataset_name +"_run_" + run_name)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    torch.save(model.state_dict(), os.path.join(path_out, "model.pt"))

    # Save dataset details as yaml file
    dataset_details_path = os.path.join(path_out, "dataset_details.yaml")
    with open(dataset_details_path, 'w') as file:
        yaml.dump(dataset_details, file)

    # Save model details as yaml file
    model_details_path = os.path.join(path_out, "model_details.yaml")
    with open(model_details_path, 'w') as file:
        yaml.dump(model_details, file)

def get_config_details(config_path, config_name):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config[config_name]
    