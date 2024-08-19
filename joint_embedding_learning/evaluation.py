import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet

from joint_embedding_learning.data.datasets import get_all_contrastive_datasets
from joint_embedding_learning.utils.utils import get_config_details
from joint_embedding_learning.utils.models_setup import get_trained_model
from joint_embedding_learning.utils.analysis_tools import apply_tsne, plot_tsne

class LogisticRegression3L(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression3L, self).__init__()
        hidden_size = 256
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        return self.model(x)

def inference_stl_10(loader, contrastive_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in tqdm(enumerate(loader)):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = contrastive_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

def inference_tactile(loader, contrastive_model, device):
    feature_vector = []
    labels_vector = []
    thetas_vector = []
    x_vector = []
    y_vector = []
    print('Inference Steps: ', len(loader))
    for step, ((x, _), (y, theta_0, x_0, y_0)) in tqdm(enumerate(loader)):
        x = x.to(device)

        # get encoding
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            h, _, z, _ = contrastive_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        thetas_vector.extend(theta_0.numpy())
        x_vector.extend(x_0.numpy())
        y_vector.extend(y_0.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    thetas_vector = np.array(thetas_vector)
    x_vector = np.array(x_vector)
    y_vector = np.array(y_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, (labels_vector, thetas_vector, x_vector, y_vector)

def inference_tactile_inv(loader, contrastive_model, device):
    feature_vector = []
    labels_vector = []
    thetas_vector = []
    x_vector = []
    y_vector = []
    print('Inference Steps: ', len(loader))
    for step, ((_, x), (y, theta_0, x_0, y_0)) in tqdm(enumerate(loader)):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = contrastive_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        thetas_vector.extend(theta_0.numpy())
        x_vector.extend(x_0.numpy())
        y_vector.extend(y_0.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    thetas_vector = np.array(thetas_vector)
    x_vector = np.array(x_vector)
    y_vector = np.array(y_vector)
    # import pdb; pdb.set_trace()
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, (labels_vector, thetas_vector, x_vector, y_vector)


def inference_tactile_visual_all(loader, contrastive_model, device):
    image_0 = []
    image_1 = []
    feature_vector_0 = []
    feature_vector_1 = []
    labels_vector = []
    for step, ((x_0, x_1), y) in tqdm(enumerate(loader)):
        x_0 = x_0.to(device)
        x_1 = x_1.to(device)

        # get encoding
        with torch.no_grad():
            h_0, _, z_0, _ = contrastive_model(x_0, x_0)
            h_1, _, z_1, _ = contrastive_model(x_1, x_1)

        h_0 = h_0.detach()
        h_1 = h_1.detach()

        image_0.extend(x_0.cpu().detach().numpy())
        image_1.extend(x_1.cpu().detach().numpy())
        feature_vector_0.extend(h_0.cpu().detach().numpy())
        feature_vector_1.extend(h_1.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector_0 = np.array(feature_vector_0)
    feature_vector_1 = np.array(feature_vector_1)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector_0.shape))
    return image_0, image_1, feature_vector_0, feature_vector_1, labels_vector

def get_features_stl_10(contrastive_model, train_loader, test_loader, dataset_type, device, save_path):
    train_X_save_path = os.path.join(save_path, 'train_X.pt')
    train_y_save_path = os.path.join(save_path, 'train_y.pt')
    test_X_save_path = os.path.join(save_path, 'test_X.pt')
    test_y_save_path = os.path.join(save_path, 'test_y.pt')

    if os.path.exists(train_X_save_path):
        train_X = torch.load(train_X_save_path)
        train_y = torch.load(train_y_save_path)
    else:
        train_X, train_y = inference_stl_10(train_loader, contrastive_model, device)
        torch.save(train_X, train_X_save_path)
        torch.save(train_y, train_y_save_path)

    if os.path.exists(test_X_save_path):
        test_X = torch.load(test_X_save_path)
        test_y = torch.load(test_y_save_path)
    else:
        test_X, test_y = inference_stl_10(test_loader, contrastive_model, device)
        torch.save(test_X, test_X_save_path)
        torch.save(test_y, test_y_save_path)
    
    return train_X, train_y, test_X, test_y

def get_features_tactile(contrastive_model, train_loader, unseen_graps_loader, unseen_tools_loader, dataset_type, device, save_path):
    train_X_bubbles_save_path = os.path.join(save_path, 'train_X_bubbles.pt')
    train_y_bubbles_save_path = os.path.join(save_path, 'train_y_bubbles.pt')
    train_X_gelslim_save_path = os.path.join(save_path, 'train_X_gelslim.pt')
    train_y_gelslim_save_path = os.path.join(save_path, 'train_y_gelslim.pt')
    unseen_graps_X_bubbles_save_path = os.path.join(save_path, 'unseen_graps_X_bubbles.pt')
    unseen_graps_y_bubbles_save_path = os.path.join(save_path, 'unseen_graps_y_bubbles.pt')
    unseen_graps_X_gelslim_save_path = os.path.join(save_path, 'unseen_graps_X_gelslim.pt')
    unseen_graps_y_gelslim_save_path = os.path.join(save_path, 'unseen_graps_y_gelslim.pt')
    unseen_tools_X_bubbles_save_path = os.path.join(save_path, 'unseen_tools_X_bubbles.pt')
    unseen_tools_y_bubbles_save_path = os.path.join(save_path, 'unseen_tools_y_bubbles.pt')
    unseen_tools_X_gelslim_save_path = os.path.join(save_path, 'unseen_tools_X_gelslim.pt')
    unseen_tools_y_gelslim_save_path = os.path.join(save_path, 'unseen_tools_y_gelslim.pt')

    if os.path.exists(train_X_bubbles_save_path):
        train_X_bubbles = torch.load(train_X_bubbles_save_path)
        train_y_bubbles = torch.load(train_y_bubbles_save_path)
    else:
        train_X_bubbles, train_y_bubbles = inference_tactile(train_loader, contrastive_model, device)
        torch.save(train_X_bubbles, train_X_bubbles_save_path)
        torch.save(train_y_bubbles, train_y_bubbles_save_path)
    
    if os.path.exists(train_X_gelslim_save_path):
        train_X_gelslim = torch.load(train_X_gelslim_save_path)
        train_y_gelslim = torch.load(train_y_gelslim_save_path)
    else:
        train_X_gelslim, train_y_gelslim = inference_tactile_inv(train_loader, contrastive_model, device)
        torch.save(train_X_gelslim, train_X_gelslim_save_path)
        torch.save(train_y_gelslim, train_y_gelslim_save_path)
    
    if os.path.exists(unseen_graps_X_bubbles_save_path):
        unseen_graps_X_bubbles = torch.load(unseen_graps_X_bubbles_save_path)
        unseen_graps_y_bubbles = torch.load(unseen_graps_y_bubbles_save_path)
    else:
        unseen_graps_X_bubbles, unseen_graps_y_bubbles = inference_tactile(unseen_graps_loader, contrastive_model, device)
        torch.save(unseen_graps_X_bubbles, unseen_graps_X_bubbles_save_path)
        torch.save(unseen_graps_y_bubbles, unseen_graps_y_bubbles_save_path)
    
    if os.path.exists(unseen_graps_X_gelslim_save_path):
        unseen_graps_X_gelslim = torch.load(unseen_graps_X_gelslim_save_path)
        unseen_graps_y_gelslim = torch.load(unseen_graps_y_gelslim_save_path)
    else:
        unseen_graps_X_gelslim, unseen_graps_y_gelslim = inference_tactile_inv(unseen_graps_loader, contrastive_model, device)
        torch.save(unseen_graps_X_gelslim, unseen_graps_X_gelslim_save_path)
        torch.save(unseen_graps_y_gelslim, unseen_graps_y_gelslim_save_path)

    if os.path.exists(unseen_tools_X_bubbles_save_path):
        unseen_tools_X_bubbles = torch.load(unseen_tools_X_bubbles_save_path)
        unseen_tools_y_bubbles = torch.load(unseen_tools_y_bubbles_save_path)
    else:
        unseen_tools_X_bubbles, unseen_tools_y_bubbles = inference_tactile(unseen_tools_loader, contrastive_model, device)
        torch.save(unseen_tools_X_bubbles, unseen_tools_X_bubbles_save_path)
        torch.save(unseen_tools_y_bubbles, unseen_tools_y_bubbles_save_path)
    
    if os.path.exists(unseen_tools_X_gelslim_save_path):
        unseen_tools_X_gelslim = torch.load(unseen_tools_X_gelslim_save_path)
        unseen_tools_y_gelslim = torch.load(unseen_tools_y_gelslim_save_path)
    else:
        unseen_tools_X_gelslim, unseen_tools_y_gelslim = inference_tactile_inv(unseen_tools_loader, contrastive_model, device)
        torch.save(unseen_tools_X_gelslim, unseen_tools_X_gelslim_save_path)
        torch.save(unseen_tools_y_gelslim, unseen_tools_y_gelslim_save_path)
        
    return train_X_bubbles, train_y_bubbles, unseen_graps_X_bubbles, unseen_graps_y_bubbles, unseen_tools_X_bubbles, unseen_tools_y_bubbles, train_X_gelslim, train_y_gelslim, unseen_graps_X_gelslim, unseen_graps_y_gelslim, unseen_tools_X_gelslim, unseen_tools_y_gelslim

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False, drop_last=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader


def train(loader, model, criterion, optimizer, device):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def test(loader, model, criterion, device):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        # import pdb; pdb.set_trace()
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

def test_visual(test_dataset, contrastive_model, model, dataset_details, save_path, device):
    visual_dataset = Subset(test_dataset, range(0, 32))
    visual_loader = torch.utils.data.DataLoader(visual_dataset, batch_size=len(visual_dataset), shuffle=False, drop_last=True)

    image_0, image_1, feature_vector_0, feature_vector_1, labels_vector = inference_tactile_visual_all(visual_loader, contrastive_model, device)
    labels_vector = torch.tensor(labels_vector).to(device)
    image_0_torch = torch.from_numpy(np.array(image_0))
    image_1_torch = torch.from_numpy(np.array(image_1))
    image_0_grid = make_grid(image_0_torch, nrow=4, normalize=True)
    image_1_grid = make_grid(image_1_torch, nrow=4, normalize=True)

    plt.imshow(image_0_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('bubbles_img')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'bubbles_img.png'))

    plt.imshow(image_1_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('gelslim_diff')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gelslim_diff.png'))

    output_0 = model(torch.tensor(np.array(feature_vector_0)).to(device))
    prediction_0 = output_0.argmax(1)
    output_1 = model(torch.tensor(np.array(feature_vector_1)).to(device))
    prediction_1 = output_1.argmax(1)

    tool_names = dataset_details['tools']['training_tools']

    print("\n")
    for i in range(len(image_0)):
        print('Image_num:', i)
        print(f"Image 0: {tool_names[labels_vector[i]]} - Prediction 0: {tool_names[prediction_0[i]]}")
        print(f"Image 1: {tool_names[labels_vector[i]]} - Prediction 1: {tool_names[prediction_1[i]]}")
        print("\n")
    
    # Save results to a CSV file
    csv_data = []
    for i in range(len(image_0)):
        row = [i, tool_names[labels_vector[i]], tool_names[prediction_0[i]], tool_names[prediction_1[i]]]
        csv_data.append(row)
    
    csv_file = os.path.join(save_path, 'visual_results.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_num', 'Label', 'Prediction_0', 'Prediction_1'])
        writer.writerows(csv_data)
    
    print("Results saved to CSV file: {}".format(csv_file))
    
    acc_0 = (prediction_0 == labels_vector).sum().item() / labels_vector.size(0)
    acc_1 = (prediction_1 == labels_vector).sum().item() / labels_vector.size(0)

    print(f"Accuracy 0: {acc_0}")
    print(f"Accuracy 1: {acc_1}")
    return

def logistic_regression_STL(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_loader, test_loader, test_dataset, dataset_type, device):
    ## Logistic Regression
    n_classes = 10
    encoder = get_resnet('resnet50', pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    model = LogisticRegression(n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', dataset_type, model_name, run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        contrastive_model, train_loader, test_loader, dataset_type, device, save_path = save_path
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, logistic_batch_size
    )

    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )

    # Save final losses and accuracies to a CSV file
    csv_file = os.path.join(save_path, 'results.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        train_loss_epoch, train_accuracy_epoch = test(args, arr_train_loader, model, criterion)
        train_loss = train_loss_epoch / len(arr_train_loader)
        train_accuracy = train_accuracy_epoch / len(arr_train_loader)
        test_loss_epoch, test_accuracy_epoch = test(args, arr_test_loader, model, criterion)
        test_loss = test_loss_epoch / len(arr_test_loader)
        test_accuracy = test_accuracy_epoch / len(arr_test_loader)
        writer.writerow([train_loss, train_accuracy, test_loss, test_accuracy])
    
    print("Loss on train set: {}".format(train_loss))
    print("Loss on test set: {}".format(test_loss))
    print("Accuracy on train set: {}".format(train_accuracy))
    print("Accuracy on test set: {}".format(test_accuracy))
    
    return

def logistic_regression_tactile(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_features, test_features, test_dataset, dataset_type, dataset_details, device):
    ## Logistic Regression
    n_classes = dataset_details['n_classes_training']
    # import pdb; pdb.set_trace()
    encoder = get_resnet('resnet50', pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    model = LogisticRegression(n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', dataset_type, model_name, run_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_X_bubbles, train_y_bubbles, train_X_gelslim, train_y_gelslim = train_features
    test_X_bubbles, test_y_bubbles, test_X_gelslim, test_y_gelslim = test_features
    arr_train_bubbles_loader, arr_test_bubbles_loader = create_data_loaders_from_arrays(train_X_bubbles, train_y_bubbles, test_X_bubbles, test_y_bubbles, logistic_batch_size)
    arr_train_gelslim_loader, arr_test_gelslim_loader = create_data_loaders_from_arrays(train_X_gelslim, train_y_gelslim, test_X_gelslim, test_y_gelslim, logistic_batch_size)

    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_bubbles_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_bubbles_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_bubbles_loader)}"
        )

    # Save final losses and accuracies to a CSV file
    csv_file = os.path.join(save_path, 'results.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Train Bubbles Loss', 'Train Bubbles Accuracy', 'Train Gelslim Loss', 'Train Gelslim Accuracy', 'Test Bubbles Loss', 'Test Bubbles Accuracy', 'Test Gelslim Loss', 'Test Gelslim Accuracy'])

        train_bubbles_loss_epoch, train_bubbles_accuracy_epoch = test(args, arr_train_bubbles_loader, model, criterion)
        train_bubbles_loss = train_bubbles_loss_epoch / len(arr_train_bubbles_loader)
        train_bubbles_accuracy = train_bubbles_accuracy_epoch / len(arr_train_bubbles_loader)

        train_gelslim_loss_epoch, train_gelslim_accuracy_epoch = test(args, arr_train_gelslim_loader, model, criterion)
        train_gelslim_loss = train_gelslim_loss_epoch / len(arr_train_gelslim_loader)
        train_gelslim_accuracy = train_gelslim_accuracy_epoch / len(arr_train_gelslim_loader)

        test_bubbles_loss_epoch, test_bubbles_accuracy_epoch = test(args, arr_test_bubbles_loader, model, criterion)
        test_bubbles_loss = test_bubbles_loss_epoch / len(arr_test_bubbles_loader)
        test_bubbles_accuracy = test_bubbles_accuracy_epoch / len(arr_test_bubbles_loader)

        test_gelslim_loss_epoch, test_gelslim_accuracy_epoch = test(args, arr_test_gelslim_loader, model, criterion)
        test_gelslim_loss = test_gelslim_loss_epoch / len(arr_test_gelslim_loader)
        test_gelslim_accuracy = test_gelslim_accuracy_epoch / len(arr_test_gelslim_loader)

        writer.writerow([train_bubbles_loss, train_bubbles_accuracy, train_gelslim_loss, train_gelslim_accuracy, test_bubbles_loss, test_bubbles_accuracy, test_gelslim_loss, test_gelslim_accuracy])

    # Get visual results
    print("Loss on train set bubbles: {}".format(train_bubbles_loss))
    print("Loss on train set gelslims: {}".format(train_gelslim_loss))
    print("Loss on test set bubbles: {}".format(test_bubbles_loss))
    print("Loss on test set gelslims: {}".format(test_gelslim_loss))

    print("Accuracy on train set bubbles: {}".format(train_bubbles_accuracy))
    print("Accuracy on train set gelslims: {}".format(train_gelslim_accuracy))
    print("Accuracy on test set bubbles: {}".format(test_bubbles_accuracy))
    print("Accuracy on test set gelslims: {}".format(test_gelslim_accuracy))
    
    test_visual(test_dataset, contrastive_model, model, dataset_details, save_path, device)

    return

def evaluation_tactile(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_loader, test_loader, test_dataset, dataset_type, dataset_details, device):
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', dataset_type, model_name, run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Create features from contrastive model
    print("### Creating features from pre-trained context model ###")
    (train_X_bubbles, train_y_bubbles, test_X_bubbles, test_y_bubbles, train_X_gelslim, train_y_gelslim, test_X_gelslim, test_y_gelslim) = get_features(
        contrastive_model, train_loader, test_loader, dataset_type, device, save_path = save_path
    )

    # import pdb; pdb.set_trace()
    # Logistic Regression Evaluation
    test_features = (test_X_bubbles, test_y_bubbles, test_X_gelslim, test_y_gelslim)
    test_len = len(test_X_bubbles)
    # train_features = (train_X_bubbles, train_y_bubbles, train_X_gelslim, train_y_gelslim)
    train_features = (train_X_bubbles[:5000], train_y_bubbles[:5000], train_X_gelslim[:test_len], train_y_gelslim[:test_len])
    
    logistic_regression_tactile(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_features, test_features, test_dataset, dataset_type, dataset_details, device)
    
    return

def evaluation(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_loader, test_loader, test_dataset, dataset_type, dataset_details, device):
    if dataset_type == 'stl_10':
        logistic_regression_STL(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_loader, test_loader, test_dataset, dataset_type, device)
    elif dataset_type == 'tactile':
        evaluation_tactile(logistic_batch_size, logistic_epochs, contrastive_model, model_name, run_name, train_loader, test_loader, test_dataset, dataset_type, dataset_details, device)
    else:
        raise NotImplementedError
    return

def get_features_TSNE_stl_10(train_X, test_X, save_path):
    tsne_train_X_save_path = os.path.join(save_path, 'tsne_train_X.pt')
    tsne_test_X_save_path = os.path.join(save_path, 'tsne_test_X.pt')

    if os.path.exists(tsne_train_X_save_path):
        tsne_train_X = torch.load(tsne_train_X_save_path)
        tsne_test_X = torch.load(tsne_test_X_save_path)
    else:
        embeddings = np.concatenate((train_X, test_X), axis=0)
        tsne_embeddings = apply_tsne(embeddings)
        tsne_train_X = tsne_embeddings[:len(train_X)]
        tsne_test_X = tsne_embeddings[len(train_X):]
        torch.save(tsne_train_X, tsne_train_X_save_path)
        torch.save(tsne_test_X, tsne_test_X_save_path)
    
    return tsne_train_X, tsne_test_X

def get_features_TSNE_tactile(train_X_bubbles, train_X_gelslim, unseen_graps_X_bubbles, unseen_graps_X_gelslim, unseen_tools_X_bubbles, unseen_tools_X_gelslim, save_path):
    tsne_train_X_bubbles_save_path = os.path.join(save_path, 'tsne_train_X_bubbles.pt')
    tsne_train_X_gelslim_save_path = os.path.join(save_path, 'tsne_train_X_gelslim.pt')
    tsne_unseen_grasps_X_bubbles_save_path = os.path.join(save_path, 'tsne_unseen_grasps_X_bubbles.pt')
    tsne_unseen_grasps_X_gelslim_save_path = os.path.join(save_path, 'tsne_unseen_grasps_X_gelslim.pt')
    tsne_unseen_tools_X_bubbles_save_path = os.path.join(save_path, 'tsne_unseen_tools_X_bubbles.pt')
    tsne_unseen_tools_X_gelslim_save_path = os.path.join(save_path, 'tsne_unseen_tools_X_gelslim.pt')

    if os.path.exists(tsne_train_X_bubbles_save_path):
        tsne_train_X_bubbles = torch.load(tsne_train_X_bubbles_save_path)
        tsne_train_X_gelslim = torch.load(tsne_train_X_gelslim_save_path)
        tsne_unseen_grasps_X_bubbles = torch.load(tsne_unseen_grasps_X_bubbles_save_path)
        tsne_unseen_grasps_X_gelslim = torch.load(tsne_unseen_grasps_X_gelslim_save_path)
        tsne_unseen_tools_X_bubbles = torch.load(tsne_unseen_tools_X_bubbles_save_path)
        tsne_unseen_tools_X_gelslim = torch.load(tsne_unseen_tools_X_gelslim_save_path)
    else:
        train_embeddings = np.concatenate((train_X_bubbles, train_X_gelslim), axis=0)
        unseen_grasps_embeddings = np.concatenate((unseen_graps_X_bubbles, unseen_graps_X_gelslim), axis=0)
        unseen_tools_embeddings = np.concatenate((unseen_tools_X_bubbles, unseen_tools_X_gelslim), axis=0)

        tsne_embeddings = apply_tsne(np.concatenate((train_embeddings, unseen_grasps_embeddings, unseen_tools_embeddings), axis=0))

        tsne_train_X_bubbles = tsne_embeddings[:len(train_X_bubbles)]
        tsne_train_X_gelslim = tsne_embeddings[len(train_X_bubbles):len(train_X_bubbles) + len(train_X_gelslim)]
        tsne_unseen_grasps_X_bubbles = tsne_embeddings[len(train_X_bubbles) + len(train_X_gelslim):len(train_X_bubbles) + len(train_X_gelslim) + len(unseen_graps_X_bubbles)]
        tsne_unseen_grasps_X_gelslim = tsne_embeddings[len(train_X_bubbles) + len(train_X_gelslim) + len(unseen_graps_X_bubbles):len(train_X_bubbles) + len(train_X_gelslim) + len(unseen_graps_X_bubbles) + len(unseen_graps_X_gelslim)]
        tsne_unseen_tools_X_bubbles = tsne_embeddings[len(train_X_bubbles) + len(train_X_gelslim) + len(unseen_graps_X_bubbles) + len(unseen_graps_X_gelslim):len(train_X_bubbles) + len(train_X_gelslim) + len(unseen_graps_X_bubbles) + len(unseen_graps_X_gelslim) + len(unseen_tools_X_bubbles)]
        tsne_unseen_tools_X_gelslim = tsne_embeddings[len(train_X_bubbles) + len(train_X_gelslim) + len(unseen_graps_X_bubbles) + len(unseen_graps_X_gelslim) + len(unseen_tools_X_bubbles):]
        
        torch.save(tsne_train_X_bubbles, tsne_train_X_bubbles_save_path)
        torch.save(tsne_train_X_gelslim, tsne_train_X_gelslim_save_path)
        torch.save(tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_bubbles_save_path)
        torch.save(tsne_unseen_grasps_X_gelslim, tsne_unseen_grasps_X_gelslim_save_path)
        torch.save(tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_bubbles_save_path)
        torch.save(tsne_unseen_tools_X_gelslim, tsne_unseen_tools_X_gelslim_save_path)
    return tsne_train_X_bubbles, tsne_train_X_gelslim, tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim

def plot_TSNE_stl_10(tsne_train_X, tsne_test_X, train_y, test_y, class_names, save_path):
    plt.figure(figsize=(10, 10))
    plot_tsne(tsne_train_X, train_y)
    plt.title('TSNE embeddings of training data')
    plt.legend(class_names)
    plt.savefig(os.path.join(save_path, 'tsne_train_plot.png'))
    plt.figure(figsize=(10, 10))
    plot_tsne(tsne_test_X, test_y)
    plt.title('TSNE embeddings of test data')
    plt.legend(class_names)
    plt.savefig(os.path.join(save_path, 'tsne_test_plot.png'))
    return

def plot_TSNE_tactile(tsne_train_X_bubbles, tsne_train_X_gelslim, tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim, dataset_details, save_path):
    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_train_X_bubbles, tsne_train_X_gelslim), axis=0), train_tool)
    plt.title('TSNE embeddings of training data - Tools')
    plt.legend(dataset_details['tools']['training_tools'])
    plt.savefig(os.path.join(save_path, 'tsne_train_tools_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_train_X_bubbles, tsne_train_X_gelslim), axis=0), train_sensor)
    plt.title('TSNE embeddings of training data - Sensors')
    plt.legend(['Bubbles', 'Gelslims'])
    plt.savefig(os.path.join(save_path, 'tsne_train_sensors_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_train_X_bubbles, tsne_train_X_gelslim), axis=0), train_thetas)
    plt.title('TSNE embeddings of training data - Thetas')
    plt.legend(np.unique(train_thetas))
    plt.savefig(os.path.join(save_path, 'tsne_train_thetas_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_train_X_bubbles, tsne_train_X_gelslim), axis=0), train_x)
    plt.title('TSNE embeddings of training data - X')
    plt.legend(np.unique(train_x))
    plt.savefig(os.path.join(save_path, 'tsne_train_x_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_train_X_bubbles, tsne_train_X_gelslim), axis=0), train_y)
    plt.title('TSNE embeddings of training data - Y')
    plt.legend(np.unique(train_y))
    plt.savefig(os.path.join(save_path, 'tsne_train_y_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim), axis=0), np.concatenate((unseen_grasp_tool, unseen_tools_tool + len(dataset_details['tools']['training_tools']))))
    plt.title('TSNE embeddings of unseen data - Tools')
    plt.legend(dataset_details['tools']['training_tools'] + dataset_details['tools']['test_tools'])
    plt.savefig(os.path.join(save_path, 'tsne_unseen_data_tools_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim), axis=0), np.concatenate((unseen_grasps_sensor, unseen_tools_sensor)))
    plt.title('TSNE embeddings of unseen data - Sensors')
    plt.legend(['Bubbles', 'Gelslims'])
    plt.savefig(os.path.join(save_path, 'tsne_unseen_data_sensors_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim), axis=0), np.concatenate((unseen_grasp_thetas, unseen_tools_thetas)))
    plt.title('TSNE embeddings of unseen data - Thetas')
    plt.legend(np.unique(np.concatenate((unseen_grasp_thetas, unseen_tools_thetas))))
    plt.savefig(os.path.join(save_path, 'tsne_unseen_data_thetas_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim), axis=0), np.concatenate((unseen_grasp_x, unseen_tools_x)))
    plt.title('TSNE embeddings of unseen data - X')
    plt.legend(np.unique(np.concatenate((unseen_grasp_x, unseen_tools_x))))
    plt.savefig(os.path.join(save_path, 'tsne_unseen_data_x_plot.png'))

    plt.figure(figsize=(10, 10))
    plot_tsne(np.concatenate((tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim), axis=0), np.concatenate((unseen_grasp_y, unseen_tools_y)))
    plt.title('TSNE embeddings of unseen data - Y')
    plt.legend(np.unique(np.concatenate((unseen_grasp_y, unseen_tools_y))))
    plt.savefig(os.path.join(save_path, 'tsne_unseen_data_y_plot.png'))
    return

def classification_task_stl_10(train_X, train_y, test_X, test_y, n_classes, logistic_epochs, device):
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.logistic_batch_size)

    encoder = get_resnet('resnet50', pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    model = LogisticRegression(n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = train(arr_train_loader, model, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}")

    # Save final losses and accuracies to a CSV file
    csv_file = os.path.join(save_path, 'results.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        train_loss_epoch, train_accuracy_epoch = test(arr_train_loader, model, criterion, device)
        train_loss = train_loss_epoch / len(arr_train_loader)
        train_accuracy = train_accuracy_epoch / len(arr_train_loader)
        test_loss_epoch, test_accuracy_epoch = test(arr_test_loader, model, criterion, device)
        test_loss = test_loss_epoch / len(arr_test_loader)
        test_accuracy = test_accuracy_epoch / len(arr_test_loader)
        writer.writerow([train_loss, train_accuracy, test_loss, test_accuracy])
    
    print("Loss on train set: {}".format(train_loss))
    print("Loss on test set: {}".format(test_loss))
    print("Accuracy on train set: {}".format(train_accuracy))
    print("Accuracy on test set: {}".format(test_accuracy))
    return

def classification_task_tactile(classification_name, n_classes,logistic_batch_size, logistic_epochs, model_name, run_name, train_features, test_features, dataset_type, dataset_details, device, logistic_regression_depth = '1L'):
    ## Logistic Regression
    # n_classes = dataset_details['n_classes_training']
    # import pdb; pdb.set_trace()
    encoder = get_resnet('resnet50', pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    if logistic_regression_depth == '3L':
        model = LogisticRegression3L(n_features, n_classes)
    else:
        model = LogisticRegression(n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', dataset_type, model_name, run_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_X_bubbles, train_y_bubbles, train_X_gelslim, train_y_gelslim = train_features
    test_X_bubbles, test_y_bubbles, test_X_gelslim, test_y_gelslim = test_features
    # import pdb; pdb.set_trace()
    arr_train_bubbles_loader, arr_test_bubbles_loader = create_data_loaders_from_arrays(train_X_bubbles, train_y_bubbles, test_X_bubbles, test_y_bubbles, logistic_batch_size)
    arr_train_gelslim_loader, arr_test_gelslim_loader = create_data_loaders_from_arrays(train_X_gelslim, train_y_gelslim, test_X_gelslim, test_y_gelslim, logistic_batch_size)

    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            arr_train_bubbles_loader, model, criterion, optimizer, device
        )
        print(
            f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_bubbles_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_bubbles_loader)}"
        )

    # Save final losses and accuracies to a CSV file
    csv_file = os.path.join(save_path, classification_name + '_results.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Train Bubbles Loss', 'Train Bubbles Accuracy', 'Train Gelslim Loss', 'Train Gelslim Accuracy', 'Test Bubbles Loss', 'Test Bubbles Accuracy', 'Test Gelslim Loss', 'Test Gelslim Accuracy'])

        train_bubbles_loss_epoch, train_bubbles_accuracy_epoch = test(arr_train_bubbles_loader, model, criterion, device)
        train_bubbles_loss = train_bubbles_loss_epoch / len(arr_train_bubbles_loader)
        train_bubbles_accuracy = train_bubbles_accuracy_epoch / len(arr_train_bubbles_loader)

        train_gelslim_loss_epoch, train_gelslim_accuracy_epoch = test(arr_train_gelslim_loader, model, criterion, device)
        train_gelslim_loss = train_gelslim_loss_epoch / len(arr_train_gelslim_loader)
        train_gelslim_accuracy = train_gelslim_accuracy_epoch / len(arr_train_gelslim_loader)

        test_bubbles_loss_epoch, test_bubbles_accuracy_epoch = test(arr_test_bubbles_loader, model, criterion, device)
        test_bubbles_loss = test_bubbles_loss_epoch / len(arr_test_bubbles_loader)
        test_bubbles_accuracy = test_bubbles_accuracy_epoch / len(arr_test_bubbles_loader)

        test_gelslim_loss_epoch, test_gelslim_accuracy_epoch = test(arr_test_gelslim_loader, model, criterion, device)
        test_gelslim_loss = test_gelslim_loss_epoch / len(arr_test_gelslim_loader)
        test_gelslim_accuracy = test_gelslim_accuracy_epoch / len(arr_test_gelslim_loader)

        writer.writerow([train_bubbles_loss, train_bubbles_accuracy, train_gelslim_loss, train_gelslim_accuracy, test_bubbles_loss, test_bubbles_accuracy, test_gelslim_loss, test_gelslim_accuracy])

    # Get visual results
    print("Loss on train set bubbles: {}".format(train_bubbles_loss))
    print("Loss on train set gelslims: {}".format(train_gelslim_loss))
    print("Loss on test set bubbles: {}".format(test_bubbles_loss))
    print("Loss on test set gelslims: {}".format(test_gelslim_loss))

    print("Accuracy on train set bubbles: {}".format(train_bubbles_accuracy))
    print("Accuracy on train set gelslims: {}".format(train_gelslim_accuracy))
    print("Accuracy on test set bubbles: {}".format(test_bubbles_accuracy))
    print("Accuracy on test set gelslims: {}".format(test_gelslim_accuracy))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument("--dataset_name", default="dataset_2", type=str)
    parser.add_argument("--model_name", default="simclr", type=str)
    parser.add_argument("--run_name", default="run_proper_logging", type=str)
    parser.add_argument("--logistic_batch_size", default=256, type=int)
    parser.add_argument("--logistic_epochs", default=500, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

   # Dataset configuration
    dataset_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'datasets_config.yaml')
    dataset_details = get_config_details(dataset_config, args.dataset_name)
    dataset_type = dataset_details['dataset_type']

    # Model configuration
    model_config =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'models_config.yaml')
    model_details = get_config_details(model_config, args.model_name)

    # Load pre-trained contrastive model from checkpoint
    contrastive_model = get_trained_model(args.model_name, model_details, dataset_type, args.run_name, args.device)
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', dataset_type, args.model_name, args.run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if dataset_type == 'stl_10':
        # Get datasets and loaders
        train_dataset, _, test_dataset = get_all_contrastive_datasets(args.dataset_name, data_split='test')

        train_dataset = ConcatDataset(train_dataset)
        test_dataset = ConcatDataset(test_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.logistic_batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.logistic_batch_size, shuffle=False, drop_last=True)

        # Save embeddings and labels
        print("### Creating features from pre-trained context model ###")
        (train_X, train_y, test_X, test_y) = get_features_stl_10(contrastive_model, train_loader, test_loader, dataset_type, args.device, save_path = save_path)

        # Save TSNE embeddings
        print("### Creating TSNE embeddings ###")
        tsne_train_X, tsne_test_X = get_features_TSNE_stl_10(train_X, test_X, save_path)

        # Plot TSNE embeddings
        print("### Plotting TSNE embeddings ###")
        plot_TSNE_stl_10(tsne_train_X, tsne_test_X, train_y, test_y, dataset_details['class_names'], save_path)

        # Classification task
        classification_task_stl_10(train_X, train_y, test_X, test_y, dataset_details['n_classes_training'], args.logistic_epochs, args.device)

    elif dataset_type == 'tactile':
        # Get datasets and loaders
        train_dataset, unseen_grasps_dataset, unseen_tools_dataset = get_all_contrastive_datasets(args.dataset_name, data_split='test')

        train_dataset = ConcatDataset(train_dataset)
        unseen_grasps_dataset = ConcatDataset(unseen_grasps_dataset)
        unseen_tools_dataset = ConcatDataset(unseen_tools_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.logistic_batch_size, shuffle=True, drop_last=True)
        unseen_grasps_loader = torch.utils.data.DataLoader(unseen_grasps_dataset, batch_size=args.logistic_batch_size, shuffle=False, drop_last=True)
        unseen_tools_loader = torch.utils.data.DataLoader(unseen_tools_dataset, batch_size=args.logistic_batch_size, shuffle=False, drop_last=True)

        # Save embeddings and labels
        print("### Creating features from pre-trained context model ###")
        train_X_bubbles, train_y_bubbles, unseen_graps_X_bubbles, unseen_graps_y_bubbles, unseen_tools_X_bubbles, unseen_tools_y_bubbles, train_X_gelslim, train_y_gelslim, unseen_graps_X_gelslim, unseen_graps_y_gelslim, unseen_tools_X_gelslim, unseen_tools_y_gelslim = get_features_tactile(contrastive_model, train_loader, unseen_grasps_loader, unseen_tools_loader, dataset_type, args.device, save_path = save_path)
        # import pdb; pdb.set_trace()
        train_sensor = np.concatenate((np.zeros(len(train_X_bubbles)), np.ones(len(train_X_gelslim))), axis=0)
        train_tool = np.concatenate((train_y_bubbles[0], train_y_gelslim[0]), axis=0)
        train_thetas = np.concatenate((train_y_bubbles[1], train_y_gelslim[1]), axis=0)
        train_x = np.concatenate((train_y_bubbles[2], train_y_gelslim[2]), axis=0)
        train_y = np.concatenate((train_y_bubbles[3], train_y_gelslim[3]), axis=0)
        unseen_grasps_sensor = np.concatenate((np.zeros(len(unseen_graps_X_bubbles)), np.ones(len(unseen_graps_X_gelslim))), axis=0)
        unseen_grasp_tool = np.concatenate((unseen_graps_y_bubbles[0], unseen_graps_y_gelslim[0]), axis=0)
        unseen_grasp_thetas = np.concatenate((unseen_graps_y_bubbles[1], unseen_graps_y_gelslim[1]), axis=0)
        unseen_grasp_x = np.concatenate((unseen_graps_y_bubbles[2], unseen_graps_y_gelslim[2]), axis=0)
        unseen_grasp_y = np.concatenate((unseen_graps_y_bubbles[3], unseen_graps_y_gelslim[3]), axis=0)
        unseen_tools_sensor = np.concatenate((np.zeros(len(unseen_tools_X_bubbles)), np.ones(len(unseen_tools_X_gelslim))), axis=0)
        unseen_tools_tool = np.concatenate((unseen_tools_y_bubbles[0], unseen_tools_y_gelslim[0]), axis=0)
        unseen_tools_thetas = np.concatenate((unseen_tools_y_bubbles[1], unseen_tools_y_gelslim[1]), axis=0)
        unseen_tools_x = np.concatenate((unseen_tools_y_bubbles[2], unseen_tools_y_gelslim[2]), axis=0)
        unseen_tools_y = np.concatenate((unseen_tools_y_bubbles[3], unseen_tools_y_gelslim[3]), axis=0)

        # Save TSNE embeddings
        tsne_train_X_bubbles, tsne_train_X_gelslim, tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim = get_features_TSNE_tactile(train_X_bubbles, train_X_gelslim, unseen_graps_X_bubbles, unseen_graps_X_gelslim, unseen_tools_X_bubbles, unseen_tools_X_gelslim, save_path)

        # Plot TSNE embeddings
        plot_TSNE_tactile(tsne_train_X_bubbles, tsne_train_X_gelslim, tsne_unseen_grasps_X_bubbles, tsne_unseen_grasps_X_gelslim, tsne_unseen_tools_X_bubbles, tsne_unseen_tools_X_gelslim, dataset_details, save_path)

        # Classification task on training tools
        classification_name = 'training_tools_classification'
        classification_task_tactile(classification_name, dataset_details['n_classes_training'], args.logistic_batch_size, args.logistic_epochs, args.model_name, args.run_name, (train_X_bubbles[:5000], train_y_bubbles[0][:5000], train_X_gelslim[:5000], train_y_gelslim[0][:5000]), (unseen_graps_X_bubbles, unseen_graps_y_bubbles[0], unseen_graps_X_gelslim, unseen_graps_y_gelslim[0]), dataset_type, dataset_details, args.device)

        # Classification task on testing tools
        classification_name = 'testing_tools_classification'
        classification_task_tactile(classification_name, dataset_details['n_classes_test'], args.logistic_batch_size, args.logistic_epochs, args.model_name, args.run_name, (unseen_tools_X_bubbles, unseen_tools_y_bubbles[0], unseen_tools_X_gelslim, unseen_tools_y_gelslim[0]), (unseen_tools_X_bubbles, unseen_tools_y_bubbles[0], unseen_tools_X_gelslim, unseen_tools_y_gelslim[0]), dataset_type, dataset_details, args.device)

        # # Classification task on training tool for thetas
        # classification_name = 'training_thetas_classification'
        # unique_values = np.unique(train_y_bubbles[1])
        # value_to_int = {value: i for i, value in enumerate(unique_values)}
        # # import pdb; pdb.set_trace()
        # tool_idx = 4
        # train_X_bubbles_train_tool = train_X_bubbles[np.where(train_y_bubbles[0] == tool_idx)]
        # train_y_bubbles_train_tool = np.array([value_to_int[value] for value in train_y_bubbles[1][np.where(train_y_bubbles[0] == tool_idx)]])
        # train_X_gelslim_train_tool = train_X_gelslim[np.where(train_y_gelslim[0] == tool_idx)] 
        # train_y_gelslim_train_tool = np.array([value_to_int[value] for value in train_y_gelslim[1][np.where(train_y_gelslim[0] == tool_idx)]])
        # # train_y_gelslim_train_tool = train_y_bubbles_train_tool

        # classification_task_tactile(classification_name, 21, 32, 500, args.model_name, args.run_name, (train_X_bubbles_train_tool, train_y_bubbles_train_tool, train_X_gelslim_train_tool, train_y_gelslim_train_tool), (train_X_bubbles_train_tool, train_y_bubbles_train_tool, train_X_gelslim_train_tool, train_y_gelslim_train_tool), dataset_type, dataset_details, args.device, logistic_regression_depth = '1L')
