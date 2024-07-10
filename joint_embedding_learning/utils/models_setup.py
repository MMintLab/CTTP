# from joint_embedding_learning.simclr.training import train as simclr_train
# from joint_embedding_learning.barlow_twins.training import train as barlow_twins_train
from joint_embedding_learning.barlow_twins.barlow import BarlowTwins
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet

import torch
import os


# def get_trainer(model_name):
#     if model_name == 'simclr':
#         return simclr_train
#     if model_name == 'barlow_twins':
#         return barlow_twins_train
#     else:
#         raise ValueError(f"Model {model_name} not found")

def get_model(model_name, model_details, pretrained=False):
    if model_name == 'simclr':
        encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder, model_details['projection_dim'], n_features)
        return model
    
    elif model_name == 'barlow_twins':
        model = BarlowTwins(model_details['projector'], pretrained=pretrained)
        return model
        
    else:
        raise ValueError(f"Model {model_name} not found")

def get_trained_model(model_name, model_details, dataset_type, run_name, device):
    if model_name == 'simclr':
        
        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)
        
        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))
        
        model = model.to(device)
        model.eval()

        return model
    
    elif model_name == 'barlow_twins':

        if run_name == 'pretrained':
            model = get_model(model_name, model_details, pretrained=True)

        elif run_name == 'random_init':
            model = get_model(model_name, model_details, pretrained=False)
        
        else:
            model = get_model(model_name, model_details)
            model_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', dataset_type, model_name, run_name, 'model.pt')
            model.load_state_dict(torch.load(model_fp, map_location=device))
            
        model = model.to(device)
        model.eval()

        return model
        
    else:
        raise ValueError(f"Model {model_name} not found")