# -*- coding: utf-8 -*-
from utils_try import resBlock, read_data, upsample
import time
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class UP(object):

    def __init__(self, sess, image_size = 40, label_size = 40, batch_size = 64, 
                 c_dim = 1, checkpoint_dir = None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        
        self.images_jp = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels')                
        
        
        self.pred = self.model()
        #self.pred = self.edsr()
        
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.labels, self.pred))
        
        tf.summary.scalar("loss",self.loss)
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=10)

    def train(self, Config):
        
        data_dir = os.path.join('./{}'.format(Config.checkpoint_dir), Config.data_dir) 
        
        train_data_jp, train_label = read_data(data_dir, Config)        
        
        self.train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)

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
            
            permutation = np.random.permutation(train_data_jp.shape[0])

            minn = 10000
            for idx in range(0, batch_idxs):
                batch_images_jp = train_data_jp[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels = train_label[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                
                counter += 1
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images_jp: batch_images_jp, self.labels: batch_labels})

                if counter % 100 == 0:
                    summary = self.sess.run(self.merged, feed_dict={self.images_jp: batch_images_jp, self.labels: batch_labels})

                    summary_writer.add_summary(summary, counter)
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                             % ((ep+1), counter, time.time()-start_time, err))
                
                if counter % 5000 == 0:
                   self.save(Config.checkpoint_dir, counter)
                if err <= minn:
                    minn = err
                    self.save(Config.checkpoint_dir, counter)
            self.save(Config.checkpoint_dir, counter)
        

    def model(self):
        
        b1_conv_low1 = tf.contrib.layers.conv2d(self.images_jp, 64, kernel_size=(3,3), stride=1, padding='SAME')
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
        output = b4_residual + self.images_jp
        
        return output
    
    def edsr(self):
        
        features_num = 128

        x = slim.conv2d(self.images, features_num, [3,3])
        conv_1 = x	
        
        for i in range(32):
            x = resBlock(x, features_num, scale=0.1)
        
        x = slim.conv2d(x, features_num, [3,3])
        x += conv_1	
        x = upsample(x, 2, features_num, None)
        x = slim.conv2d(x, 1, [3,3], activation_fn=None)
        output = x
        
        return output

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