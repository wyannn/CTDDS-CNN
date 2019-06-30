# -*- coding: utf-8 -*-

from utils import read_data, dct2, idct2, upsample
import time
import os
import tensorflow as tf
import numpy as np

class TRY(object):

    def __init__(self, sess, image_size = 41, label_size = 41, batch_size = 64,
                 c_dim = 1, checkpoint_dir = None, training = True, scale=3):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.training = training
        self.scale = scale

        self.build_model()

    def build_model(self):
        
        self.images_jp = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels')

        self.weights_unit = {   
                
                'conv1_1': tf.get_variable(name='conv1_1', shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),
                'conv1_2': tf.get_variable(name='conv1_2', shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),           
                
                'conv2_1': tf.get_variable(name='conv2_1', shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),
                'conv2_2': tf.get_variable(name='conv2_2', shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),           
                  
                'conv3_1': tf.get_variable(name='conv3_1', shape=[5,5,64,64], initializer=tf.contrib.layers.xavier_initializer()),
                'conv3_2': tf.get_variable(name='conv3_2', shape=[5,5,64,64], initializer=tf.contrib.layers.xavier_initializer()),           
                
                'conv4_1': tf.get_variable(name='conv4_1', shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),
                'conv4_2': tf.get_variable(name='conv4_2', shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),      

                }
    
        self.biases_unit = {   
                
                'conv1_1': tf.Variable(tf.zeros([64]), name='conv1_1'),       
                'conv1_2': tf.Variable(tf.zeros([64]), name='conv1_2'), 
                
                'conv2_1': tf.Variable(tf.zeros([64]), name='conv2_1'),       
                'conv2_2': tf.Variable(tf.zeros([64]), name='conv2_2'), 

                'conv3_1': tf.Variable(tf.zeros([64]), name='conv3_1'),       
                'conv3_2': tf.Variable(tf.zeros([64]), name='conv3_2'),  
                
                'conv4_1': tf.Variable(tf.zeros([64]), name='conv4_1'),       
                'conv4_2': tf.Variable(tf.zeros([64]), name='conv4_2'),

                }
        
 
        #mae
        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.labels, self.pred))
        
        #dct
        #self.pred, self.pred2, self.pred3 = self.model()
        #self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.pred2, self.pred3))
        #self.loss = tf.reduce_mean(tf.squared_difference(self.pred2, self.pred3))

        #mse
        #self.pred = self.model()
        #self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.pred)) 
        
        #self.pred1, self.pred2, self.pred3 = self.model()
        #self.loss = (tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred1)) 
        #            + tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred2))
        #            + tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred3)))
        
        #regularization
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #reg_constant = 1e-5
        #self.loss = self.loss + reg_constant * sum(reg_losses)
        
        #self.pred1, self.pred2, self.pred3 = self.model()
        #self.loss = (tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred1))        
        #            + tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred2))
        #            + tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred3)))
        
        
        """
        #mse and l1
        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.labels - self.images, self.pred))
        
        self.l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1e-4)
        self.weights = tf.trainable_variables() 
        for w in self.weights:
            if len(w.shape) < 2:
                self.weights.remove(w)
    
        self.regularization_penalty = tf.contrib.layers.apply_regularization(self.l1_regularizer, weights_list=self.weights)

        self.loss = self.loss + self.regularization_penalty 
        """
        
        
        tf.summary.scalar("loss",self.loss)
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def train(self, Config):
        
        data_dir = os.path.join('./{}'.format(Config.checkpoint_dir), Config.data_dir) #获取训练数据的地址
        
        #train_data_jp, train_data_bp, train_label = read_data(data_dir, Config) 
        train_data_jp, train_label = read_data(data_dir, Config)        
        
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)
        """
        
        self.train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)
        
        """
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(Config.learning_rate, global_step*Config.batch_size, 20*len(train_data)*Config.batch_size, 0.1, staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        """
        
        #gradient clip
        """
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(Config.learning_rate, global_step*Config.batch_size, 5*len(train_data)*Config.batch_size, 0.1, staircase=True)        
        
        opt = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
        grad_and_value = opt.compute_gradients(self.loss)
        
        clip = tf.Variable(Config.clip_grad, name='clip') 
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -clip, clip)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in grad_and_value]
        
        self.train_op = opt.apply_gradients(capped_gvs, global_step=global_step)
        """
        
        
        tf.global_variables_initializer().run()
        
        summary_writer = tf.summary.FileWriter("./graph",graph=tf.get_default_graph())
        
        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("Load SUCCESS.")
        else:
            print("Load failed!")

        print("Training...")

        for ep in range(Config.epoch):
            batch_idxs = len(train_data_jp) // Config.batch_size
            
            #shuffle
            permutation = np.random.permutation(train_data_jp.shape[0])
            #train_data = train_data[permutation,:, :, :]
            #train_label = train_label[permutation,:, :, :]
            minn = 10000
            for idx in range(0, batch_idxs):
                batch_images_jp = train_data_jp[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                #batch_images_bp = train_data_bp[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels = train_label[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                
                #permutation = np.random.choice(train_data.shape[0], Config.batch_size)
                #batch_images = train_data[permutation,:, :, :]
                #batch_labels = train_label[permutation,:, :, :]

                counter += 1
                #_, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images_jp: batch_images_jp, self.images_bp: batch_images_bp, self.labels: batch_labels})
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images_jp: batch_images_jp, self.labels: batch_labels})

                if counter % 100 == 0:
                    #summary = self.sess.run(self.merged, feed_dict={self.images_jp: batch_images_jp, self.images_bp: batch_images_bp, self.labels: batch_labels})
                    summary = self.sess.run(self.merged, feed_dict={self.images_jp: batch_images_jp, self.labels: batch_labels})

                    summary_writer.add_summary(summary, counter)
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                             % ((ep+1), counter, time.time()-start_time, err))
                
                if counter % 10000 == 0:
                   self.save(Config.checkpoint_dir, counter)
                if err <= minn:
                    minn = err
                    self.save(Config.checkpoint_dir, counter)
            self.save(Config.checkpoint_dir, counter)
            #if ep == 9 or ep == 19 or ep == 29 or ep == 39 or ep == 49 or ep == 59 or ep == 69 or ep == 79:
            #    self.save(Config.checkpoint_dir, counter)
        
                
                    

    def model(self):

        #
        b1_conv_low1 = tf.contrib.layers.conv2d(self.images_jp, 32, kernel_size=(3,3), stride=1, padding='SAME')
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

        #
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

        #
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
        output = b4_residual + self.images_jp

        return output

    def model_bp(self):

        #
        b1_conv_low1 = tf.contrib.layers.conv2d(self.images, 32, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv_low2 = tf.contrib.layers.conv2d(b1_conv_low1, 64, kernel_size=(3, 3), stride=1, padding='SAME')

        b1_conv1_1 = tf.contrib.layers.conv2d(b1_conv_low2, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv1_2 = tf.contrib.layers.conv2d(b1_conv1_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv1_add = b1_conv_low2 + 0.05 * b1_conv1_2

        b1_conv2_1 = tf.contrib.layers.conv2d(b1_conv1_add, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv2_2 = tf.contrib.layers.conv2d(b1_conv2_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv2_add = b1_conv1_add + 0.05 * b1_conv2_2

        b1_conv3_1 = tf.contrib.layers.conv2d(b1_conv2_add, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv3_2 = tf.contrib.layers.conv2d(b1_conv3_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b1_conv3_add = b1_conv2_add + 0.05 * b1_conv3_2

        b1_total = tf.concat([b1_conv2_add, b1_conv3_add], 3)
        b1_mpool = tf.layers.max_pooling2d(b1_total, 2, strides=2, padding='SAME')
        b1_apool = tf.layers.average_pooling2d(b1_mpool,
                                               [b1_mpool.get_shape().as_list()[1], b1_mpool.get_shape().as_list()[2]],
                                               strides=1)
        b1_squeeze = tf.contrib.layers.conv2d(b1_apool, 16, kernel_size=(1, 1), stride=1, padding='SAME',
                                              activation_fn=tf.sigmoid)
        b1_extract = tf.contrib.layers.conv2d(b1_squeeze, b1_total.get_shape().as_list()[-1], kernel_size=(1, 1),
                                              stride=1, padding='SAME', activation_fn=tf.sigmoid)
        b1_scale = b1_total * b1_extract

        #
        b2_conv_dr1 = tf.contrib.layers.conv2d(b1_scale, 64, kernel_size=(1, 1), stride=1, padding='SAME',
                                               activation_fn=None)
        b2_conv_dr2 = tf.contrib.layers.conv2d(b2_conv_dr1, 64, kernel_size=(3, 3), stride=1, padding='SAME')

        b2_conv1_1 = tf.contrib.layers.conv2d(b2_conv_dr2, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b2_conv1_2 = tf.contrib.layers.conv2d(b2_conv1_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b2_conv1_add = b2_conv_dr2 + 0.05 * b2_conv1_2

        b2_conv2_1 = tf.contrib.layers.conv2d(b2_conv1_add, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b2_conv2_2 = tf.contrib.layers.conv2d(b2_conv2_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b2_conv2_add = b2_conv1_add + 0.05 * b2_conv2_2

        b2_rconv1_1 = tf.nn.relu(
            tf.nn.conv2d(b2_conv2_add, self.weights_unit['conv1_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv1_1'])
        b2_rconv1_2 = tf.nn.relu(
            tf.nn.conv2d(b2_rconv1_1, self.weights_unit['conv1_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv1_2'])
        b2_rconv1_add = b2_conv2_add + 0.1 * b2_rconv1_2

        b2_rconv2_1 = tf.nn.relu(
            tf.nn.conv2d(b2_rconv1_add, self.weights_unit['conv1_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv1_1'])
        b2_rconv2_2 = tf.nn.relu(
            tf.nn.conv2d(b2_rconv2_1, self.weights_unit['conv1_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv1_2'])
        b2_rconv2_add = b2_conv2_add + 0.1 * b2_rconv2_2

        b2_rconv3_1 = tf.nn.relu(
            tf.nn.conv2d(b2_rconv2_add, self.weights_unit['conv1_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv1_1'])
        b2_rconv3_2 = tf.nn.relu(
            tf.nn.conv2d(b2_rconv3_1, self.weights_unit['conv1_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv1_2'])
        b2_rconv3_add = b2_conv2_add + 0.1 * b2_rconv3_2

        b2_total = tf.concat([b1_total, b2_conv1_add, b2_conv2_add, b2_rconv1_add, b2_rconv2_add, b2_rconv3_add], 3)
        b2_mpool = tf.layers.max_pooling2d(b2_total, 2, strides=2, padding='SAME')
        b2_apool = tf.layers.average_pooling2d(b2_mpool,
                                               [b2_mpool.get_shape().as_list()[1], b2_mpool.get_shape().as_list()[2]],
                                               strides=1)
        b2_squeeze = tf.contrib.layers.conv2d(b2_apool, 16, kernel_size=(1, 1), stride=1, padding='SAME',
                                              activation_fn=tf.sigmoid)
        b2_extract = tf.contrib.layers.conv2d(b2_squeeze, b2_total.get_shape().as_list()[-1], kernel_size=(1, 1),
                                              stride=1, padding='SAME', activation_fn=tf.sigmoid)
        b2_scale = b2_total * b2_extract

        #
        b3_conv_low1_1 = tf.contrib.layers.conv2d(self.images, 32, kernel_size=(3, 3), stride=1, padding='SAME')
        b3_conv_low1_2 = tf.contrib.layers.conv2d(b3_conv_low1_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')

        b3_3conv1_1 = tf.nn.relu(
            tf.nn.conv2d(b3_conv_low1_2, self.weights_unit['conv2_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv2_1'])
        b3_3conv1_2 = tf.nn.relu(
            tf.nn.conv2d(b3_3conv1_1, self.weights_unit['conv2_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv2_2'])
        b3_3conv1_add = b3_conv_low1_2 + 0.1 * b3_3conv1_2

        b3_3conv2_1 = tf.nn.relu(
            tf.nn.conv2d(b3_3conv1_add, self.weights_unit['conv2_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv2_1'])
        b3_3conv2_2 = tf.nn.relu(
            tf.nn.conv2d(b3_3conv2_1, self.weights_unit['conv2_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv2_2'])
        b3_3conv2_add = b3_conv_low1_2 + 0.1 * b3_3conv2_2

        b3_3conv3_1 = tf.nn.relu(
            tf.nn.conv2d(b3_3conv2_add, self.weights_unit['conv2_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv2_1'])
        b3_3conv3_2 = tf.nn.relu(
            tf.nn.conv2d(b3_3conv3_1, self.weights_unit['conv2_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv2_2'])
        b3_3conv3_add = b3_conv_low1_2 + 0.1 * b3_3conv3_2

        b3_5conv1_1 = tf.nn.relu(
            tf.nn.conv2d(b3_conv_low1_2, self.weights_unit['conv3_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv3_1'])
        b3_5conv1_2 = tf.nn.relu(
            tf.nn.conv2d(b3_5conv1_1, self.weights_unit['conv3_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv3_2'])
        b3_5conv1_add = b3_conv_low1_2 + 0.1 * b3_5conv1_2

        b3_5conv2_1 = tf.nn.relu(
            tf.nn.conv2d(b3_5conv1_add, self.weights_unit['conv3_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv3_1'])
        b3_5conv2_2 = tf.nn.relu(
            tf.nn.conv2d(b3_5conv2_1, self.weights_unit['conv3_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv3_2'])
        b3_5conv2_add = b3_conv_low1_2 + 0.1 * b3_5conv2_2

        b3_5conv3_1 = tf.nn.relu(
            tf.nn.conv2d(b3_5conv2_add, self.weights_unit['conv3_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv3_1'])
        b3_5conv3_2 = tf.nn.relu(
            tf.nn.conv2d(b3_5conv3_1, self.weights_unit['conv3_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv3_2'])
        b3_5conv3_add = b3_conv_low1_2 + 0.1 * b3_5conv3_2

        b3_combine = tf.concat([b3_3conv3_add, b3_5conv3_add], 3)
        b3_mask = tf.contrib.layers.conv2d(b3_combine, 64, kernel_size=(3, 3), stride=1, padding='SAME',
                                           activation_fn=tf.sigmoid)

        """
        #
        b5_conv_low1 = tf.contrib.layers.conv2d(self.images_jph, 32, kernel_size=(7,7), stride=1, padding='SAME')
        b5_conv_low2 = tf.contrib.layers.conv2d(b5_conv_low1, 64, kernel_size=(7,7), stride=1, padding='SAME')

        b5_rconv1_1 = tf.nn.relu(tf.nn.conv2d(b5_conv_low2, self.weights_unit['conv5_1'], strides=[1,1,1,1], padding='SAME') + self.biases_unit['conv5_1'])
        b5_rconv1_2 = tf.nn.relu(tf.nn.conv2d(b5_rconv1_1, self.weights_unit['conv5_2'], strides=[1,1,1,1], padding='SAME') + self.biases_unit['conv5_2'])
        b5_rconv1_add = b5_conv_low2 + 0.1 * b5_rconv1_2 

        b5_rconv2_1 = tf.nn.relu(tf.nn.conv2d(b5_rconv1_add, self.weights_unit['conv5_1'], strides=[1,1,1,1], padding='SAME') + self.biases_unit['conv5_1'])
        b5_rconv2_2 = tf.nn.relu(tf.nn.conv2d(b5_rconv2_1, self.weights_unit['conv5_2'], strides=[1,1,1,1], padding='SAME') + self.biases_unit['conv5_2'])
        b5_rconv2_add = b5_conv_low2 + 0.1 * b5_rconv2_2 

        b5_rconv3_1 = tf.nn.relu(tf.nn.conv2d(b5_rconv2_add, self.weights_unit['conv5_1'], strides=[1,1,1,1], padding='SAME') + self.biases_unit['conv5_1'])
        b5_rconv3_2 = tf.nn.relu(tf.nn.conv2d(b5_rconv3_1, self.weights_unit['conv5_2'], strides=[1,1,1,1], padding='SAME') + self.biases_unit['conv5_2'])
        b5_rconv3_add = b5_conv_low2 + 0.1 * b5_rconv3_2 

        b5_output = tf.contrib.layers.conv2d(b5_rconv3_add, 1, kernel_size=(7,7), stride=1, padding='SAME', activation_fn=None)
        """

        #
        b4_conv_dr1 = tf.contrib.layers.conv2d(b2_scale, 64, kernel_size=(1, 1), stride=1, padding='SAME',
                                               activation_fn=None)
        b4_conv_dr2 = tf.contrib.layers.conv2d(b4_conv_dr1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b4_conv_dr2 = b4_conv_dr2

        b4_conv1_1 = tf.contrib.layers.conv2d(b4_conv_dr2, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b4_conv1_2 = tf.contrib.layers.conv2d(b4_conv1_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b4_conv1_add = b4_conv_dr2 + 0.05 * b4_conv1_2

        b4_conv2_1 = tf.contrib.layers.conv2d(b4_conv1_add, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b4_conv2_2 = tf.contrib.layers.conv2d(b4_conv2_1, 64, kernel_size=(3, 3), stride=1, padding='SAME')
        b4_conv2_add = b4_conv1_add + 0.05 * b4_conv2_2

        b4_rconv1_1 = tf.nn.relu(
            tf.nn.conv2d(b4_conv2_add, self.weights_unit['conv4_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv4_1'])
        b4_rconv1_2 = tf.nn.relu(
            tf.nn.conv2d(b4_rconv1_1, self.weights_unit['conv4_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv4_2'])
        b4_rconv1_add = b4_conv2_add + 0.1 * b4_rconv1_2

        b4_rconv2_1 = tf.nn.relu(
            tf.nn.conv2d(b4_rconv1_add, self.weights_unit['conv4_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv4_1'])
        b4_rconv2_2 = tf.nn.relu(
            tf.nn.conv2d(b4_rconv2_1, self.weights_unit['conv4_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv4_2'])
        b4_rconv2_add = b4_conv2_add + 0.1 * b4_rconv2_2

        b4_rconv3_1 = tf.nn.relu(
            tf.nn.conv2d(b4_rconv2_add, self.weights_unit['conv4_1'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv4_1'])
        b4_rconv3_2 = tf.nn.relu(
            tf.nn.conv2d(b4_rconv3_1, self.weights_unit['conv4_2'], strides=[1, 1, 1, 1], padding='SAME') +
            self.biases_unit['conv4_2'])
        b4_rconv3_add = b4_conv2_add + 0.1 * b4_rconv3_2

        b4_enhance = b4_rconv3_add * b3_mask
        b4_residual = tf.contrib.layers.conv2d(b4_enhance, 1, kernel_size=(3, 3), stride=1, padding='SAME',
                                               activation_fn=None)
        b4_output = b4_residual + self.images

        return b4_output

    def save(self, checkpoint_dir, step):
        model_name = "TRY.model"
        model_dir = "%s_%s" % ("try", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print("Reading checkpoints...")
        model_dir = "%s_%s" % ("try", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False