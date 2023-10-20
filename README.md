# Enhancing Fake News Detection in Social Media via Label Propagation on Cross-modal Tweet Graph

[toc]

## 1. Introduction

This repository provides the code for our paper submitted to ACM MM 2023.

This project contains all the engineering files involved in Enhancing Fake News Detection in Social Media via Label Propagation on Cross-modal Tweet Graph, implementing a complete process of CLIP-based cross-modal tweet graph construction as well as training and testing the model.

## File Structure 

- `CLIP_Graph_Construction.ipynb`: This Jupyter notebook contains the code for constructing the graph structure based on the CLIP model.
- `LPN_layer.py`: This Python script contains the implementation of the LPN (Label Propagation Network) layer used in the model.
- `model.py`: This Python script defines the overall FCN-LP model architecture.
- `Training_and_Testing.ipynb`: This Jupyter notebook contains the code for training and testing the model.
- `dataset/`: This folder contains the dataset used for the project, but is not available due to size constraints in the supplementary materials..

## Usage Instructions

1. **Environment Setup**: To set up the environment, you will need Python 3.8 and the following libraries:
    - numpy
    - pandas
    - torch
    - torchvision
    - tqdm
	- torch_geometric
	- https://github.com/openai/CLIP.git
	
2. **Dataset Preparation**: Place your dataset files in the `dataset/` folder. Make sure the dataset is in the correct format, as required by the code.

3. **Graph Construction**: Run the `CLIP_Graph_Construction.ipynb` notebook to construct the graph structure using the CLIP model. This will create and save the cross-modal tweet graph, which will be used later in the training and testing process.

4. **Training and Testing**: Run the `Training_and_Testing.ipynb` notebook to train and test our model. This notebook will load the graph and tweet feature created in the previous step, perform training using the provided dataset, and then test the model's performance.

