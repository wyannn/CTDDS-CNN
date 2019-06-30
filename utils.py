# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:12:04 2017

@author: student
"""
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.fftpack
 
def read_data(path, Config):

    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        #data2 = np.array(hf.get('data2'))
        label = np.array(hf.get('label'))
        data = np.reshape(data, [data.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        #data2 = np.reshape(data2, [data2.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        label = np.reshape(label, [label.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        return data, label


def  maxmin(data, ymin, ymax):

    xmax = tf.reduce_max(data)
    xmin = tf.reduce_min(data)
    res = (ymax - ymin) * (data - xmin) / (xmax - xmin) + ymin
    return res


def dct2(image): 
    imsize = image.get_shape().as_list()
    dctmtx8 = tf.constant([[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274],
                                    [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.0975451610080642, -0.0975451610080641, -0.277785116509801, -0.415734806151273, -0.490392640201615],
                                    [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643, -0.461939766255643, -0.191341716182545, 0.191341716182545, 0.461939766255643],
                                    [0.415734806151273, -0.0975451610080641, -0.490392640201615, -0.277785116509801, 0.277785116509801, 0.490392640201615, 0.0975451610080640, -0.415734806151272],
                                    [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274, 0.353553390593274, -0.353553390593273, -0.353553390593274, 0.353553390593273],
                                    [0.277785116509801, -0.490392640201615, 0.0975451610080642, 0.415734806151273, -0.415734806151273, -0.0975451610080649, 0.490392640201615, -0.277785116509801],
                                    [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545, -0.191341716182545, 0.461939766255644, -0.461939766255644, 0.191341716182543],
                                    [0.0975451610080642, -0.277785116509801, 0.415734806151273, -0.490392640201615, 0.490392640201615, -0.415734806151272, 0.277785116509802, -0.0975451610080625]]) 
    dctmtx8_t = tf.transpose(dctmtx8)
    dctmtx8 = tf.reshape(dctmtx8, [1, 8, 8])
    dctmtx8_t = tf.reshape(dctmtx8_t, [1, 8, 8])
    dctmtx8 = tf.tile(dctmtx8, [imsize[0] * (imsize[1]//8) * (imsize[2]//8), 1, 1]) 
    dctmtx8_t = tf.tile(dctmtx8_t, [imsize[0] * (imsize[1]//8) * (imsize[2]//8), 1, 1]) 
    image_dct = tf.matmul(tf.matmul(dctmtx8, image), dctmtx8_t)
    image_dct = tf.reshape(image_dct, [-1, 8, 8, 1])
    
    return image_dct

def idct2(image_dct): 
    imsize = image_dct.get_shape().as_list()
    dctmtx8 = tf.constant([[0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274],
                            [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.0975451610080642, -0.0975451610080641, -0.277785116509801, -0.415734806151273, -0.490392640201615],
                            [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643, -0.461939766255643, -0.191341716182545, 0.191341716182545, 0.461939766255643],
                            [0.415734806151273, -0.0975451610080641, -0.490392640201615, -0.277785116509801, 0.277785116509801, 0.490392640201615, 0.0975451610080640, -0.415734806151272],
                            [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274, 0.353553390593274, -0.353553390593273, -0.353553390593274, 0.353553390593273],
                            [0.277785116509801, -0.490392640201615, 0.0975451610080642, 0.415734806151273, -0.415734806151273, -0.0975451610080649, 0.490392640201615, -0.277785116509801],
                            [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545, -0.191341716182545, 0.461939766255644, -0.461939766255644, 0.191341716182543],
                            [0.0975451610080642, -0.277785116509801, 0.415734806151273, -0.490392640201615, 0.490392640201615, -0.415734806151272, 0.277785116509802, -0.0975451610080625]]) 
    i_dctmtx8 = tf.matrix_inverse(dctmtx8)
    dctmtx8_t = tf.transpose(dctmtx8)
    i_dctmtx8_t = tf.matrix_inverse(dctmtx8_t)
    i_dctmtx8 = tf.reshape(i_dctmtx8, [1, 8, 8])
    i_dctmtx8_t = tf.reshape(i_dctmtx8_t, [1, 8, 8])
    i_dctmtx8 = tf.tile(i_dctmtx8, [imsize[0] * (imsize[1]//8) * (imsize[2]//8), 1, 1]) 
    i_dctmtx8_t = tf.tile(i_dctmtx8_t, [imsize[0] * (imsize[1]//8) * (imsize[2]//8), 1, 1]) 
    image_idct = tf.matmul(tf.matmul(i_dctmtx8, image_dct), i_dctmtx8_t)
    image_idct = tf.reshape(image_idct, [-1, 8, 8, 1])
    
    return image_idct

def extract_patches(x):
    return tf.extract_image_patches(
        x,
        (1, 8, 8, 1),
        (1, 8, 8, 1),
        (1, 1, 1, 1),
        padding="VALID"
    )

def extract_patches_inverse(x, y):
    _x = tf.zeros_like(x)
    _y = extract_patches(_x)
    grad = tf.gradients(_y, _x)[0]

    return tf.gradients(_y, _x, grad_ys=y)[0] / grad

            

def upsample(x,scale=2,features=64,activation=tf.nn.relu):
	assert scale in [2,3,4]
	x = slim.conv2d(x,features,[3,3],activation_fn=activation)
	if scale == 2:
		ps_features = (scale**2)
		x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
		x = PS(x,2,color=False)
	elif scale == 3:
		ps_features =3*(scale**2)
		x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		#x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x,3,color=False)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x,2,color=False)
	return x

"""
def _phase_shift(I, r):
	bsize, a, b, c = I.get_shape().as_list()
	bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
	X = tf.reshape(I, (bsize, a, b, r, r))
	X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
	X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
	X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
	return tf.reshape(X, (bsize, a*r, b*r, 1))
"""
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

def res_mod_layers(in_data, num_filters, kernel_size, strides, padding, is_training):
    
    # Batch Norm
    bn_out = tf.layers.batch_normalization(
        inputs=in_data,
        scale=False,
        training=is_training)
    # ReLU
    act_out = tf.nn.relu(bn_out)
    # conv
    conv_out = tf.layers.conv2d(
        inputs=act_out,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding)

    return conv_out

def un_max_pool(net,mask,stride):

    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret