# install enviroment according to install_image_recognition.sh
# for script usage please activate enviroment:
# conda activate for_image_recognition
# python <this_script>.py

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.

"""
Natalia Czub
"""

from datetime import date
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2, os, gc, glob
from tqdm import tqdm

import tensorflow.keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator


# MAY BE CHANGED

# PARAMETERS
# size of images min 75
resize = 175
# epochs
gridsearch_nn_epochs = 1
final_epochs = 2
# batch size
batch_size = 64
# batch size for data augmentation
data_augmentation = False
batch_size_generator = 5
# percentage of test set 
test_size = 0.10
# validation_size is 11% of training set, after splitting train-test, which is about 10% of whole dataset
validation_size = 0.11
random_state = 42
# learning rate
lr=0.001
# optimizers: SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad, Adamax, Adafactor, Nadam, Ftrl
optimizer = 'adam'
# number of classes for classification model
number_of_classes = 2
# gridsearch through neuralnetworks: option True or False, if False you need to choose which neural network will be basic model
NN_gridsearch = True
base_model_chosen = 'Xception'


final_dropout = 0.1

# name of folder with pictures - the name of folder is one category
class_0 = "Viral_Pneumonia"
class_1 = "Normal"

path_of_images = r'/data1/dane/image_recognition/ankle_classification/github_20_09_2023'


# DO NOT CHANGE
os.listdir(path_of_images)

imagePaths = []
for dirname, _, filenames in os.walk(path_of_images):
    for filename in filenames:
        if (filename[-3:] == 'png'):
            imagePaths.append(os.path.join(dirname, filename))
            
           

todays_date = date.today()
folder_name = f"result_{todays_date}_dataAug_{data_augmentation}_NNgrids_{NN_gridsearch}_imgsize_{resize}_epochs_{final_epochs}_batchsize_{batch_size}"
os.mkdir(folder_name)

Data = []
Target = []
cat = {class_0: class_0, class_1: class_1}

for imagePath in tqdm(imagePaths):
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (resize, resize)) /255

    Data.append(image)
    Target.append(cat[label])

    
df = pd.DataFrame(Target,columns=['Labels'])
plt.figure(figsize=(8,6))
sx=sns.countplot(x ='Labels', data = df,palette="crest")
sx.set_xticklabels(labels=sx.get_xticklabels())
plt.savefig(f"{folder_name}/amount_of_classes.png")

print(f'{class_1}:',Target.count(f'{class_1}'))
print(f'{class_0}:',Target.count(f'{class_0}'))


plt.figure(figsize=(20,12))
for n , i in enumerate(list(np.random.randint(0,len(imagePaths),12))) : 
    plt.subplot(2,6,n+1)
    plt.imshow(Data[i] , cmap='gray')
    plt.title(Target[i])
    plt.axis('off')     

plt.savefig(f"{folder_name}/example_of_images_size_{resize}.png")    

le = LabelEncoder()
labels = le.fit_transform(Target)
labels = to_categorical(labels)


(x_train, x_test, y_train, y_test) = train_test_split(Data, labels,test_size=test_size ,
                                                      stratify=labels,random_state=random_state)

(x_train_main, x_validation, y_train_main, y_validation) = train_test_split(x_train, y_train, test_size=validation_size,
                                                      stratify=y_train,random_state=random_state)

# CHANGES OF IMAGES' SIZE
s = resize
trainX = np.array(x_train_main)
testX = np.array(x_test)
trainY = np.array(y_train_main)
testY = np.array(y_test)
validationX = np.array(x_validation)
validationY = np.array(y_validation)

print("train_X shape:", trainX.shape)
print("test_X shape:", testX.shape)
print("train_Y shape", trainY.shape)
print("test_Y shape", testY.shape)
print("validation_X shape", validationX.shape)
print("validation_Y shape", validationY.shape)


# DATA AUGMENTATION
train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode="nearest")

train_generator = train_datagen.flow(trainX, trainY, batch_size=batch_size_generator)

# MODEL CREATION - neural networks limited by keras distribution
# as next step add NNs like: googlenet, alexnet, lenet, 
model_dict = {'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
              'InceptionV3': tf.keras.applications.InceptionV3,
              'Xception': tf.keras.applications.Xception,
              'ResNet50': tf.keras.applications.ResNet50,
              'ResNet101': tf.keras.applications.ResNet101,
              'ResNet152': tf.keras.applications.ResNet152,
              'ResNet50V2': tf.keras.applications.ResNet50V2,
              'ResNet101V2': tf.keras.applications.ResNet101V2,
              'ResNet152V2': tf.keras.applications.ResNet152V2,
              'MobileNet': tf.keras.applications.MobileNet,
              'MobileNetV2': tf.keras.applications.MobileNetV2,
              'MobileNetV3Small': tf.keras.applications.MobileNetV3Small,
              'MobileNetV3Large': tf.keras.applications.MobileNetV3Large,
              'DenseNet121': tf.keras.applications.densenet.DenseNet121,
             'DenseNet169': tf.keras.applications.densenet.DenseNet169,
              'DenseNet201': tf.keras.applications.densenet.DenseNet201,
             'EfficientNetB1': tf.keras.applications.EfficientNetB1,
             'EfficientNetB2': tf.keras.applications.EfficientNetB2, 
             'EfficientNetB3': tf.keras.applications.EfficientNetB3, 
             'EfficientNetB4': tf.keras.applications.EfficientNetB4,
              'EfficientNetB5': tf.keras.applications.EfficientNetB5, 
              'EfficientNetB6': tf.keras.applications.EfficientNetB6,
              'EfficientNetB7': tf.keras.applications.EfficientNetB7,
              'VGG16': tf.keras.applications.VGG16,
              'VGG19': tf.keras.applications.VGG19,
              'NASNetLarge': tf.keras.applications.NASNetLarge,
              'NASNetMobile': tf.keras.applications.NASNetMobile,
              'ConvNeXtTiny': tf.keras.applications.ConvNeXtTiny,
              'ConvNeXtSmall': tf.keras.applications.ConvNeXtSmall,
              'ConvNeXtBase': tf.keras.applications.ConvNeXtBase,
              'ConvNeXtLarge': tf.keras.applications.ConvNeXtLarge,
              'ConvNeXtXLarge': tf.keras.applications.ConvNeXtXLarge
             }

if NN_gridsearch is True:
    name_of_model = []
    val_loss_values = []
    val_accuracy_values = []
    history_dict = {}
    for model_name, model_class in model_dict.items():
        print(f"Training {model_name}...")
        model = model_class(include_top=False, input_shape=(resize, resize, 3))
        x = model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(number_of_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=model.input, outputs=predictions)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        if data_augmentation is True:
            history = model.fit(train_generator, epochs=gridsearch_nn_epochs, 
                                validation_data= (validationX, validationY))
        else:
            history = model.fit(trainX, trainY, epochs=gridsearch_nn_epochs, 
                                validation_data= (validationX, validationY))
        history_dict[model_name] = history
        name_of_model.append(model_name)
        val_loss_value = history.history['val_loss'][-1]
        val_loss_values.append(val_loss_value)
        val_accuracy_value = history.history['val_accuracy'][-1]
        val_accuracy_values.append(val_accuracy_value)
    df_name_model = pd.DataFrame(name_of_model)
    df_val_loss_value = pd.DataFrame(val_loss_values)
    df_val_accuracy = pd.DataFrame(val_accuracy_values)
    df_summary = pd.concat([df_name_model, df_val_loss_value, df_val_accuracy], axis = 1)
    df_summary.columns = ["name_of_NN", "val_loss", "val_accuracy"]
    df_summary.to_csv(f"{folder_name}/summary_of_NN_gridsearch.csv", index = False, header = True)

    # Find the model with the lowest validation loss
    best_model_name = min(history_dict, key=lambda k: history_dict[k].history['val_loss'][-1])
    base_model = model_dict[best_model_name](include_top=False, input_shape=(resize, resize, 3))
    model = tf.keras.Sequential([
    base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(final_dropout), tf.keras.layers.Dense(number_of_classes, activation='softmax')])
else:
        base_model = model_dict[base_model_chosen](include_top=False, input_shape=(resize, resize, 3))
        model = tf.keras.Sequential([base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(final_dropout), tf.keras.layers.Dense(number_of_classes, activation='softmax')])




model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])


print(model.summary())
with open(f"{folder_name}/modelsummary.txt", 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
# MODEL TREANING
if data_augmentation is True:
    model.fit(train_generator, epochs=final_epochs, batch_size=batch_size, verbose=1, validation_data= (validationX, validationY))
else:
    model.fit(trainX, trainY,  epochs=final_epochs, batch_size=batch_size, verbose=1, validation_data= (validationX, validationY))
save_model(model, f"{folder_name}/model_{todays_date}.h5")
# model = load_model('model.h5')

# PREDICTIONS WITH TEST SET
modelLoss, modelAccuracy = model.evaluate(testX, testY, verbose=0)

print('Test Loss is {}'.format(modelLoss))
print('Test Accuracy is {}'.format(modelAccuracy))

class_names = [f'{class_0}',f'{class_1}']
y_pred = model.predict(testX)


y_test_df = pd.DataFrame(y_test)
y_pred_df = pd.DataFrame(y_pred.round(3))
y_test_df.columns = class_names
y_test_df.columns = [str(col) + '_true' for col in y_test_df.columns]

y_pred_df.columns = class_names
y_pred_df.columns = [str(col) + '_pred' for col in y_pred_df.columns]

predictions = pd.concat([y_test_df, y_pred_df], axis = 1)
predictions.to_csv(f"{folder_name}/test_set_predictions.csv", index = False, header = True)


plt.figure(figsize=(8,8))
x = confusion_matrix(testY.argmax(axis=1),y_pred.argmax(axis=1))
Confusion_Matrix = pd.DataFrame(x, index=class_names, columns=class_names)

sns.set(font_scale=1.5, color_codes=True, palette='deep')
sns.heatmap(Confusion_Matrix, annot=True, annot_kws={'size':10}, fmt='d', cmap="crest")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f'Confusion Matrix  - Test set - Accuracy: {round(modelAccuracy,3)}')
plt.savefig(f"{folder_name}/confusion_matrix_test_set.png")
