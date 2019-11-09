# -*- coding: utf-8 -*-
import h5py
import numpy as np
 
def read_data(path, Config):

    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        data = np.reshape(data, [data.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        label = np.reshape(label, [label.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        return data, label

