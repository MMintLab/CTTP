# CTTP
We introduce a contrastive self-supervised learning approach that represents tactile feedback across different sensor types. Our method utilizes paired tactile data—where two distinct sensors, in our case Soft Bubbles and GelSlims, grasp the same object in the same configuration—to learn a unified latent representation.

# Dataset
We use Touch2Touch Dataset.

Paper: https://www.arxiv.org/abs/2409.08269

Dataset: https://drive.google.com/drive/folders/15vWo5AWw9xVKE1wHbLhzm40ClPyRBYk5?usp=sharing

# Before installation
Install SimCLR package: https://github.com/Spijkervet/SimCLR/tree/master
```
pip install simclr
```

Install T3 package: https://github.com/alanzjl/t3/tree/341177f232df3b824a5246b0d1855cd9e4d2cf29
```
git clone https://github.com/alanzjl/t3
cd t3
pip install -e .
```

# Checkpoints
Download checkpoint to CTTP root folder: https://drive.google.com/drive/folders/1A5bRuciQ4nuSa1r1X7JB7pMYJHJ6j6vc

CTTP Model Checkpoint: https://drive.google.com/file/d/10_HR54aKSUuF3hQPTY1zLOgYuBHMITch/view?usp=drive_link

# Train CTTP Model
```
cd scripts
python train_model.py --model_name simclr --device cuda:0 --dataset dataset_1
```

# Evaluate CTTP Model
```
cd joint_embedding_learning/
python evaluation.py --dataset_name dataset_1 --run_name dataset_1_run_B_128 --model_name simclr
```
