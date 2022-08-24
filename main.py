# -*- coding: utf-8 -*-
"""
@author: Mohammad Asim
Code originally written for kaggle notebook
"""
from input_pipeline import build_dataset
from model import Model
from settings import Settings
from video_generator import generate_video
import numpy as np
from trainer import train
import pandas as pd
import tensorflow as tf 
import json
import matplotlib.pyplot as plt

def dump(history, settings):
    """
    Dump history data in csv format and settings configuration in json format
    """
    
    # Store training data
    df = pd.DataFrame.from_dict(history['train'])
    df.to_csv("train_data.csv")
    
    # Store validation data
    df = pd.DataFrame.from_dict(history['val'])
    df.to_csv("val_data.csv")
    
    # Store learning rate 
    df = pd.DataFrame.from_dict(history['lr'])
    df.to_csv("lr.csv")
    
    # Store settings object
    with open('config.json', 'w') as fp:
        json.dump(settings.dict(), fp)
        
def plot(history):
    """
    History Plotter
    """
    
    plt.subplot(2,2,1)
    plt.plot(history['train']['energy'])
    plt.subplot(2,2,2)
    plt.plot(history['train']['epe'])
    plt.subplot(2,2,3)
    plt.plot(history['lr'])
    plt.subplot(2,2,4)
    plt.plot(history['val']['epe'])
    
def main():
    settings = Settings(
        name='model.h5',
        lr=0.001, #0.0008
        lr_decay=0.8,
        patience=12,
        epochs=5,
        alpha=0.01, #0.01
        lmbda=0.005, # 0.8, 0.1, 0.08
        epsilon=10e-3, 
        multi_scale_weights=[0.2, 0.4, 0.6, 0.8, 1]
    )
    
    # Load training and validation data
    file_paths = np.sort(tf.io.gfile.glob('./dataset/train/albedo/*'))
    train_dataset = build_dataset(file_paths[0], batch_size=32, cache=True, shuffle=False)
    val_dataset = build_dataset(file_paths[8], batch_size=32, cache=True, shuffle=False)
    
    # Initialize the model
    model = Model().build_network(tf.keras.layers.LeakyReLU(), load=False)
    
    # Train and save the model
    model, history = train((train_dataset, val_dataset), model, settings)
    model.save(settings.name)
    
    # Store data and plot the results
    dump(history, settings)
    plot(history)

    # Perform prediction and generate video for prediction at scale 0
    # To generate for other scales, set the correct index for the pred
    # Make sure to pass the same dataset in the generate video argument as used for prediction
    pred = model.predict(val_dataset, verbose=1)
    generate_video(pred[0], val_dataset)
    
if __name__=="__main__":
    main()

