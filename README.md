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

# Train CTTP Model

# Evaluate CTTP Model
