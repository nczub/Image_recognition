# Image_recognition

Python script for binary classification of images. It has several options. 

First, it may search through the known Keras structure of NeuralNetworks, such as Xception, ResNet, Inception, ConvNeXt, and more. 

A system based on preliminary training gets the best architecture and trains it longer. 

You can also choose a particular architecture and train only on this one.

Moreover, you can choose if you want to use data augmentation, which may increase training dataset size, by multiplication and small changes of images.

By default, the script divides two folders of images into training, test, and validation sets (80:10:10 ratio).

As a user, you can change also other parameters such as batch_size, learning rate, and type of optimizer. 


## Installation of environment

The needed packages are in the file 'install_image_recognition.sh'.

- You need to run this script in a console. After this, a conda environment will be created 'for_image_recognition'. 

- Activation of environment:

```bash
$ conda activate for_image_recognition
```

- Run the script

```bash
python binary_classification.py
```

## Images
I copy images from 
as the examples for the GitHub project. 



