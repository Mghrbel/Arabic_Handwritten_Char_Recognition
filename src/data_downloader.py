import zipfile
import os
import shutil
import pandas as pd
import numpy as np


DATA_DIR = '../data/'

def download_dataset() :
    # Download data from kaggle api on src directory
    os.system('kaggle competitions download -c arabic-hwr-ai-pro-intake1')
    print("Data downloaded successfully")

    #unzipping data in data directory from home dir
    with zipfile.ZipFile('arabic-hwr-ai-pro-intake1.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)



def create_labeled_trained_dirs():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    characters_list = np.array(sorted(['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ','ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع','غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى']))
    

    if not os.path.isdir(os.path.join(DATA_DIR, 'train_labeled/')):
        os.mkdir(DATA_DIR + 'train_labeled/')

    for _, row in train_df.iterrows():
        id = row['id']
        l = row['label']

        if not os.path.isdir(os.path.join(DATA_DIR, 'train_labeled/{}/'.format(characters_list[l-1]))):
            os.mkdir(os.path.join(DATA_DIR, 'train_labeled/{}/'.format(characters_list[l-1])))

        shutil.copy(DATA_DIR+'train/{:05d}.png'.format(id), DATA_DIR+'train_labeled/{}/'.format(characters_list[l-1]))
    print("labeled data created successfully")