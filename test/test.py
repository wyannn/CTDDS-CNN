# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:18:35 2018

@author: student

"""

import tensorflow as tf
import os
import numpy as np
from PIL import Image

test = '../classical_v/qf_30'

pic_list = os.listdir(test)
for img in pic_list:
    if ('qf' in test and os.path.splitext(img)[1] == '.bmp'):
        image = np.array(Image.open(test + '/' + img)) / 255.
        
        input_image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
        tf.reset_default_graph()
        images = tf.placeholder(tf.float32, [1, image.shape[0], image.shape[1], 1], name='images')
        
        b1_conv_low1 = tf.contrib.layers.conv2d(images, 32, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv_low2 = tf.contrib.layers.conv2d(b1_conv_low1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        
        b1_conv1_1 = tf.contrib.layers.conv2d(b1_conv_low2, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv1_2 = tf.contrib.layers.conv2d(b1_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv1_add = b1_conv_low2 + 0.05 * b1_conv1_2
    
        b1_conv2_1 = tf.contrib.layers.conv2d(b1_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv2_2 = tf.contrib.layers.conv2d(b1_conv2_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv2_add = b1_conv1_add + 0.05 * b1_conv2_2
        
        b1_conv3_1 = tf.contrib.layers.conv2d(b1_conv2_add, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv3_2 = tf.contrib.layers.conv2d(b1_conv3_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b1_conv3_add = b1_conv2_add + 0.05 * b1_conv3_2
    
        b1_total = tf.concat([b1_conv2_add, b1_conv3_add], 3)
        
        b2_conv_dr1 = tf.contrib.layers.conv2d(b1_total, 64, kernel_size=(1,1), stride=1, padding='SAME', activation_fn=None)
        b2_conv_dr2 = tf.contrib.layers.conv2d(b2_conv_dr1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        
        b2_conv1_1 = tf.contrib.layers.conv2d(b2_conv_dr2, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b2_conv1_2 = tf.contrib.layers.conv2d(b2_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b2_conv1_add = b2_conv_dr2 + 0.05 * b2_conv1_2
        
        b2_conv2_1 = tf.contrib.layers.conv2d(b2_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b2_conv2_2 = tf.contrib.layers.conv2d(b2_conv2_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b2_conv2_add = b2_conv1_add + 0.05 * b2_conv2_2
        
        b2_conv3_1 = tf.contrib.layers.conv2d(b2_conv2_add, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b2_conv3_2 = tf.contrib.layers.conv2d(b2_conv3_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b2_conv3_add = b2_conv2_add + 0.05 * b2_conv3_2
        
        b2_total = tf.concat([b1_total, b2_conv1_add, b2_conv2_add, b2_conv3_add], 3)

        b4_conv_dr1 = tf.contrib.layers.conv2d(b2_total, 64, kernel_size=(1,1), stride=1, padding='SAME', activation_fn=None)
        b4_conv_dr2 = tf.contrib.layers.conv2d(b4_conv_dr1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        
        b4_conv1_1 = tf.contrib.layers.conv2d(b4_conv_dr2, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b4_conv1_2 = tf.contrib.layers.conv2d(b4_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b4_conv1_add = b4_conv_dr2 + 0.05 * b4_conv1_2
        
        b4_conv2_1 = tf.contrib.layers.conv2d(b4_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b4_conv2_2 = tf.contrib.layers.conv2d(b4_conv2_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b4_conv2_add = b4_conv1_add + 0.05 * b4_conv2_2 
        
        b4_conv3_1 = tf.contrib.layers.conv2d(b4_conv2_add, 64, kernel_size=(3,3), stride=1, padding='SAME')
        b4_conv3_2 = tf.contrib.layers.conv2d(b4_conv3_1, 64, kernel_size=(3,3), stride=1, padding='SAME')
        
        b4_residual = tf.contrib.layers.conv2d(b4_conv3_2, 1, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
        output = b4_residual + images
       
        checkpoint_dir = '../checkpoint/qf_30'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model_dir = os.path.join(checkpoint_dir, ckpt_name)

        saver = tf.train.Saver() 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_dir)
            results = sess.run(output, feed_dict={images: input_image})
            
    
        res = (np.squeeze(results)) * 255.
        Image.fromarray(np.uint8(res)).save('../res/res_' + img)