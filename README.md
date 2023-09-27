# Image_recognition

Python script for binary image classification. It has several options. 

First, it can search the known structures of Keras NeuralNetworks, such as Xception, ResNet, Inception, ConvNeXt, and more. 
The pre-training-based system takes the best architecture and trains it longer.

You can also choose a specific architecture and train only on it.

What's more, you can choose to use data augmentation, which can increase the size of the training dataset by multiplying and slightly changing images.

By default, the script divides two folders (=two classes) of images into training, test, and validation sets (80:10:10 ratio). You need to set the names of the folders.

You can also change other parameters, such as batch_size, learning rate, and optimizer type.


## Installation of environment

The needed packages are in the file 'install_image_recognition.sh'.

- You need to run this script in a console. After this, a conda environment will be created 'for_image_recognition'. 

- Activation of environment:

```bash
$ conda activate for_image_recognition
```

## Run the script

```bash
python binary_classification.py
```

## Images
I copy the images from  

https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

as the examples for the GitHub project. 



