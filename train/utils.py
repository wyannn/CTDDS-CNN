# -*- coding: utf-8 -*-
import h5py
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
 
def read_data(path, Config):

    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        data = np.reshape(data, [data.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        label = np.reshape(label, [label.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        return data, label

def upsample(x,scale=2,features=64,activation=tf.nn.relu):
    
    assert scale in [2,3,4]
    x = slim.conv2d(x,features,[3,3],activation_fn=activation)
    if scale == 2:
        ps_features = (scale**2)
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
        x = PS(x,2,color=False)
    elif scale == 3:
        ps_features =3*(scale**2)
        x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
        x = PS(x,3,color=False)
    elif scale == 4:
        ps_features = 3*(2**2)
        for i in range(2):
            x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
            x = PS(x,2,color=False)
    return x


def _phase_shift(I, r):
    
    return tf.depth_to_space(I, r)


def PS(X, r, color=False):
    
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc],3)
    else:
        X = _phase_shift(X, r)
    return X


def resBlock(x,channels=64,kernel_size=[3,3],scale=1):
    
    tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
    tmp *= scale
    return x + tmp