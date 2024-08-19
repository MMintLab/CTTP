import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_embeddings(results_path, split='train', sensor='bubbles', dataset = 'stl_10'):
    if dataset == 'stl_10':
        embeddings = torch.load(os.path.join(results_path, f'{split}_X.pt'))
        labels = torch.load(os.path.join(results_path, f'{split}_y.pt'))
        sensor_array = np.array([0]*len(labels))
    
    else:
        if sensor == 'both':
            g_embeddings = torch.load(os.path.join(results_path, f'{split}_X_gelslim.pt'))
            g_labels = torch.load(os.path.join(results_path, f'{split}_y_gelslim.pt'))
            b_embeddings = torch.load(os.path.join(results_path, f'{split}_X_bubbles.pt'))
            b_labels = torch.load(os.path.join(results_path, f'{split}_y_bubbles.pt'))

            embeddings = np.concatenate([g_embeddings, b_embeddings], axis=0)
            labels = np.concatenate([g_labels, b_labels], axis=0)
            sensor_array = np.array([0]*len(g_labels) + [1]*len(b_labels))
            
        else:
            embeddings = torch.load(os.path.join(results_path, f'{split}_X_{sensor}.pt'))
            labels = torch.load(os.path.join(results_path, f'{split}_y_{sensor}.pt'))
            sensor_array = np.array([0]*len(labels))

    return embeddings, labels, sensor_array

def apply_tsne(embeddings, n_components=2, perplexity=30.0, learning_rate=200.0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    return tsne.fit_transform(embeddings)

def apply_tsne_3d(embeddings, n_components=3, perplexity=30.0, learning_rate=200.0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    return tsne.fit_transform(embeddings)

def plot_tsne_tactile(tsne_embeddings, labels, colors, sensor='bubbles', sensor_vis = False):
    if sensor == 'bubbles':
        marker_style = 'o'
        alpha = 0.2
        edge = 'blue'

        if sensor_vis:
            colors = ['blue']*len(np.unique(labels))
        
    else:
        marker_style = 'x'
        alpha = 1.0
        edge = 'black'
        
        if sensor_vis:
            colors = ['black']*len(np.unique(labels))

    # plt.figure(figsize=(10, 10))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=label, alpha=alpha, marker=marker_style, color=colors[label], s=30)
 
    # plt.show()
    return

def plot_tsne(tsne_embeddings, labels, alpha=1.0, marker_style='o'):
    unique_labels = np.unique(labels)
    # colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], label=label, alpha=alpha, marker=marker_style, color=label_to_color[label], s=30)
    return