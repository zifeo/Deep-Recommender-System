# Group: Nils Bonfils, Teo Stocco, Yann Dupont-Costedoat
#Project 2: Recommender System 

## Requirements
Python3 is necessary to run the programs.
Git is necessary to install all libraries.

Libraries:
 * Scipy
 * Numpy
 * Keras v1.1.1
 * Tensorflow v0.11.0
 * h5py
 * Sklearn
 * mca
 * tqdm

## Installation 
Execute the following command to install the libraries:
pip3 install scipy keras==1.1.1 sklearn numpy h5py tqdm git+https://github.com/esafak/mca

In order to install Tensorflow, follow the guide on this link: https://www.tensorflow.org/get_started/os_setup
/!\ In the "export" command replace the version number from "12" to "11".

## Data
The dataset from Kaggle are not provided inside the ZIP archive, but they must to be stored in the "data" folder with unchanged names ("data_train.csv" and "sampleSubmission.csv")
All other files for the programs are also already in the data folder.

## run.py
This script implements the neural network producing our best predictions. It loads a previously created and trained network in order to reproduce our best submission.
Note: The ratings might differ from our Kaggle submission starting at the 7th or 8th decimal digit due to floating-point differences (GPU vs CPU/system).


## run_als.py
This script implements Alternating Least Squares


## train.ipynb
This script creates the neural network and trains it.

## setup.ipynb
This script splits and forges the data to be used by the neural network.
