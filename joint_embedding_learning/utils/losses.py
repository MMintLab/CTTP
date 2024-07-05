import torch 

def loss_vector_embedding(z_i, z_j):
    loss = torch.nn.functional.mse_loss(z_i, z_j)
    return loss

def loss_vector_similarity(h_i, h_j):
    loss = torch.nn.functional.cosine_similarity(h_i, h_j)
    return loss