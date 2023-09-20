#!/bin/bash

ENV_NAME="for_image_recognition"
PYTHON_VER="3.9"
LOG_FILE="install_image_recognition_log.txt"



rm -f $LOG_FILE

#remove old environment if it exists
conda env remove -y --name=$ENV_NAME >> $LOG_FILE 2>&1
rm -rf ~/anaconda3/envs/$ENV_NAME >> $LOG_FILE 2>&1
# create conda env
conda create -y --name=$ENV_NAME python=$PYTHON_VER >> $LOG_FILE 2>&1


# activate $ENV_NAME environment
source ~/.bashrc >> $LOG_FILE 2>&1
conda init bash >> $LOG_FILE 2>&1
conda activate $ENV_NAME >> $LOG_FILE 2>&1

conda install -y -c conda-forge cudatoolkit cudnn >> $LOG_FILE 2>&1
conda install -y -c conda-forge tqdm >> $LOG_FILE 2>&1
conda install -y -c anaconda seaborn >> $LOG_FILE 2>&1


#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


pip install --upgrade setuptools pip >> $LOG_FILE 2>&1
pip install tensorflow==2.11.0 >> $LOG_FILE 2>&1
pip install git+https://github.com/keras-team/keras-tuner.git >> $LOG_FILE 2>&1
pip install autokeras >> $LOG_FILE 2>&1
pip install scikit-learn >> $LOG_FILE 2>&1
pip install numpy==1.21.6 scipy >> $LOG_FILE 2>&1
pip install opencv-python >> $LOG_FILE 2>&1


exit 0
