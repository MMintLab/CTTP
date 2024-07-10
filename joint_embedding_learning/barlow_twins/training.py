import torch
import numpy as np
import argparse
from tqdm import tqdm
from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.utils.utils import load_optimizer, save_model
from joint_embedding_learning.utils.losses import loss_vector_embedding

from joint_embedding_learning.barlow_twins.barlow import BarlowTwins, BarlowLoss, LARS, adjust_learning_rate

def train_step(args, train_loaders, model, criterion, optimizer, scaler, wandb):
    loss_epoch = 0
    loss_vector = 0
    count = 0
    loader_len = len(train_loaders[0]) * len(train_loaders)
    step_lr = args.current_epoch * loader_len
    for loader in train_loaders:
        for step, ((x_i, x_j), _) in tqdm(enumerate(loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            adjust_learning_rate(args.epochs, args.batch_size, args.learning_rate_weights, args.learning_rate_biases, optimizer, loader_len, step_lr)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # positive pair, with encoding
                h_i, h_j, z_i, z_j = model(x_i, x_j)
                loss = criterion(z_i, z_j)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            with torch.no_grad():
                # model.eval()
                loss_vector_0 = loss_vector_embedding(h_i, h_j)
                # model.train()

            if step % 50 == 0:
                print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")
                print(f"Step [{step}/{len(loader)}]\t Loss Vector: {loss_vector_0.item()}")

            wandb.log({"Loss_train/Barlow_Loss_batch": loss.item(), "Loss_train/embedding_diff_batch": loss_vector_0.item(), "Epoch_batch": args.current_epoch})
            args.global_step += 1

            loss_epoch += loss.item()
            loss_vector += loss_vector_0.item()
            count += 1
            step_lr += 1
    
    loss_epoch /= count
    loss_vector /= count
    return loss_epoch, loss_vector, x_i, x_j

def validation_step(args, val_loaders, model, criterion, wandb, val_name = "Val"):
    loss_epoch = 0
    loss_vector = 0
    count = 0

    # import pdb; pdb.set_trace()
    for val_loader in val_loaders:
        for step, ((x_i, x_j), _) in tqdm(enumerate(val_loader)):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            # import pdb; pdb.set_trace()
            loss = criterion(z_i, z_j)
            loss_vector_0 = loss_vector_embedding(h_i, h_j)

            if step % 50 == 0:
                print(f"Step [{step}/{len(val_loader)}]\t" + val_name + "Loss: {loss.item()}")
                print(f"Step [{step}/{len(val_loader)}]\t" + val_name + "Loss Vector: {loss_vector_0.item()}")

            wandb.log({"Loss_" + val_name + "/Barlow_Loss_batch": loss.item(), "Loss_"+ val_name + "/embedding_diff_batch": loss_vector_0.item(), "Epoch": args.current_epoch})

            loss_epoch += loss.item()
            loss_vector += loss_vector_0.item()
            count += 1

    loss_epoch /= count
    loss_vector /= count
        # model.train()

    return loss_epoch, loss_vector, x_i, x_j


def train(train_datasets, validation_datasets, dataset_name, dataset_details, model_name, model_details, wandb, run_name = '', device='cuda:0'):
    model_parser = argparse.ArgumentParser(description="Model-Configuration")
    model_parser.add_argument('--dataset_name', type=str, default='dataset_2', help='Name of the dataset')
    model_parser.add_argument('--model_name', type=str, default='simclr', help='Name of the model')
    model_parser.add_argument("--device", default=device, type=str, help="Device to run the model on")
    model_parser.add_argument("--run_name", type=str, default=run_name, help='Name of the run')
    model_parser.add_argument("--debug", action='store_true', help='Debug mode')
    args = model_parser.parse_args()

    for key, value in model_details.items():
        setattr(args, key, value)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_type = dataset_details['dataset_type']
    model_type = model_name
    
    train_loaders = []
    for dataset in train_datasets:
        train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
    
    val_loaders_array = {}
    for key, value in validation_datasets.items():
        val_loaders = []
        for dataset in value:
            val_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False))
        val_loaders_array[key] = val_loaders

    # initialize model
    model = BarlowTwins(args.projector)
    model = model.to(args.device)

    # optimizer / loss
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
    
    # Criterion
    criterion = BarlowLoss(args.projector, args.batch_size, args.lambd, args.device)

    args.global_step = 0
    args.current_epoch = 0
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Epochs"):
    # for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]

        loss_epoch, loss_vector, x_i, x_j = train_step(args, train_loaders, model, criterion, optimizer, scaler, wandb)
        wandb.log({"Loss_train/Model_Loss": loss_epoch, "Loss_train/embedding_diff":loss_vector, "Misc/learning_rate": lr, "Epoch": epoch})

        with torch.no_grad():
            # model.eval()
            for key, val_loader in val_loaders_array.items():
                val_loss_epoch, val_loss_vector, x_i, x_j = validation_step(args, val_loader, model, criterion, wandb, key)
                wandb.log({"Loss_" + key + "/Barlow_Loss": val_loss_epoch, "Loss_" + key + "/embedding_diff": val_loss_vector, "Epoch": epoch})
            
            # model.train()

        if epoch % 10 == 0:
            save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=run_name)
            wandb.log({"Image": [wandb.Image(x_i[:36]), wandb.Image(x_j[:36])], "Epoch": epoch})

        args.current_epoch += 1

    save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name=run_name)