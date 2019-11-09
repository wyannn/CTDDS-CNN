# -*- coding: utf-8 -*-
from model import UP
import tensorflow as tf
import os

class Config():

    epoch = 20
    learning_rate = 1e-4
    batch_size = 64
    image_size = 40
    label_size = 40
    c_dim = 1
    checkpoint_dir = 'checkpoint'  
    data_dir = 'train.h5'

def main():

    if not os.path.exists(Config.checkpoint_dir):
        os.makedirs(Config.checkpoint_dir)

    with tf.Session() as sess:
        trysr = UP(sess,
                  image_size = Config.image_size, 
                  label_size = Config.label_size, 
                  batch_size = Config.batch_size,
                  c_dim = Config.c_dim, 
                  checkpoint_dir = Config.checkpoint_dir)

        trysr.train(Config)
    
if __name__ == '__main__':
  main()