# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import xrange
import time
import tensorflow as tf
import numpy as np
import os

from residual_net import dense_layer, residual_net, residual_block, batch_norm, conv2d, conv2d_block, conv2d_transpose, conv2d_transpose_block, lrelu
from dataset import FontDataManager
from preprocess_font import render_frame

class GAN(object):
    def __init__(self, sess, batch_size=64, image_size=64, input_channels=1, 
                 dataset_name='font', sample_size=100):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_channels = input_channels
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        
    def encoder(self, x, is_train, is_reuse=False):
        with tf.variable_scope('encoder') as scope:
            if is_reuse:
                scope.reuse_variables()            
            e_conv_1 = conv2d_block(x, is_train, filter_shape=[5, 5, self.input_channels, 64], activation=lrelu, name='e_conv_1')
            e_residual_block_1 = residual_block(e_conv_1, 1, 128, is_train=is_train, activation=lrelu, is_downsample=True, name='e_resblock_1')
            e_residual_block_2 = residual_block(e_residual_block_1, 1, 256, is_train=is_train, activation=lrelu, is_downsample=True, name='e_resblock_2')
            e_residual_block_3 = residual_block(e_residual_block_2, 1, 512, is_train=is_train, activation=lrelu, is_downsample=True, name='e_resblock_3')
        return e_residual_block_3
    
    def generator(self, x, is_train, is_reuse=False):
        with tf.variable_scope('generator') as scope:
            if is_reuse:
                scope.reuse_variables()            
            g_conv_1 = conv2d_block(x, is_train, filter_shape=[5, 5, x.get_shape()[-1], 512], activation=lrelu, name='g_conv_1')
            g_deconv_1 = conv2d_transpose_block(g_conv_1, is_train, filter_shape=[5, 5, 256, 512], activation=lrelu, name='g_deconv_1')
            g_residual_block_1 = residual_block(g_deconv_1, 1, 256, is_train, activation=lrelu, is_downsample=False, name='g_resblock_1')
            g_deconv_2 = conv2d_transpose_block(g_residual_block_1, is_train, filter_shape=[5, 5, 128, 256], activation=lrelu, name='g_deconv_2')
            g_residual_block_2 = residual_block(g_deconv_2, 1, 128, is_train, activation=lrelu, is_downsample=False, name='g_resblock_2')
            g_deconv_3 = conv2d_transpose_block(g_residual_block_2, is_train, filter_shape=[5, 5, 64, 128], activation=lrelu, name='g_deconv_3')
            g_conv_2 = conv2d_block(g_deconv_3, is_train, filter_shape=[5, 5, 64, self.input_channels], activation=tf.nn.tanh, name='g_conv_2')
        return g_conv_2
    
    def discriminator(self, x, is_train, is_reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if is_reuse:
                scope.reuse_variables()            
            d_conv_1 = conv2d_block(x, is_train, filter_shape=[5, 5, x.get_shape()[-1], 64], name='d_conv_1')
            d_residual_block_1 = residual_block(d_conv_1, 2, 128, is_train=is_train, is_downsample=True, name='d_resblock_1')
            d_residual_block_2 = residual_block(d_residual_block_1, 2, 256, is_train=is_train, is_downsample=True, name='d_resblock_2')
            d_residual_block_3 = residual_block(d_residual_block_2, 2, 512, is_train=is_train, is_downsample=True, name='d_resblock_3')
            d_conv_2 = conv2d_block(d_residual_block_3, is_train, filter_shape=[5, 5, 512, 512], name='d_conv2d_2')
            d_reshape = tf.reshape(d_conv_2, [self.batch_size, -1], name='d_reshape')
            d_output = dense_layer(d_reshape, d_reshape.get_shape()[-1], 1, activation=None, name='d_dense_layer')
        return d_output, d_reshape
    
    def bulid_model(self):        
        self.source_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_channels], name='source_images')
        self.target_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_channels], name='target_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size, self.image_size, self.image_size, self.input_channels], name='sample_images')
    
        self.E = self.encoder(self.source_images, is_train=True)
        self.G = self.generator(self.E, is_train=True)
        self.D_real, feature_real = self.discriminator(self.target_images, is_train=True)
        self.D_fake, feature_fake = self.discriminator(self.G, is_train=True, is_reuse=True)
        
        self.D_real_sum = tf.summary.histogram("d_real", self.D_real)
        self.D_fake_sum = tf.summary.histogram("d_fake", self.D_fake)
        self.G_sum = tf.summary.image("G", self.G)
    
        self.d_gan_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.d_gan_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(self.D_fake)))
        self.g_gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake, tf.ones_like(self.D_fake)))
    
        self.d_gan_loss_real_sum = tf.summary.scalar("d_gan_loss_real", self.d_gan_loss_real)
        self.d_gan_loss_fake_sum = tf.summary.scalar("d_gan_loss_fake", self.d_gan_loss_fake)
        self.g_gan_loss_sum = tf.summary.scalar("g_gan_loss_sum", self.g_gan_loss)
        
        self.d_loss = self.d_gan_loss_real + self.d_gan_loss_fake
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        self.image_loss = tf.reduce_mean(tf.square(self.G - self.target_images))
        self.feature_loss = tf.reduce_mean(tf.square(feature_real - feature_fake))
        self.g_loss = self.g_gan_loss + self.feature_loss + self.image_loss
        
        self.image_loss_sum = tf.summary.scalar("image_loss_sum", self.image_loss)
        self.feature_loss_sum = tf.summary.scalar("feature_loss", self.feature_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        
    
        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'encoder' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.eg_vars = self.e_vars + self.g_vars
        
        self.sampler = self.generator(self.encoder(self.sample_images, is_train=False, is_reuse=True), 
                                      is_train=False, is_reuse=True)        
        self.saver = tf.train.Saver()
        
    def train(self, config):
        #get data
        dataset = FontDataManager(config.source_font, config.target_font, 
                                  config.train_size, config.validation_size, unit_scale=True, shuffle=True)
        sample_soruce_image, sample_target_image= dataset.get_validation()
        
        self.bulid_model()
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.eg_vars)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        self.summary_all = tf.summary.merge_all()
        
        self.summary_writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        
        counter = 1
        batch_idxs = config.train_size // self.batch_size
        start_time = time.time()
        
        if self.load(config.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        for epoch in xrange(config.epoch):
            for idx in xrange(0, batch_idxs):
                source_batch, target_batch = dataset.next_train_batch(self.batch_size)
               
                # Update D network
                self.sess.run([d_optim], feed_dict={ self.source_images:source_batch, self.target_images:target_batch })
                
                # Update G network two times
                self.sess.run([g_optim], feed_dict={ self.source_images:source_batch, self.target_images:target_batch })
                self.sess.run([g_optim], feed_dict={ self.source_images:source_batch, self.target_images:target_batch })

                
                d_gan_loss_real, d_gan_loss_fake, d_loss, g_gan_loss, image_loss, feature_loss, g_loss = self.sess.run(\
                    [self.d_gan_loss_real, self.d_gan_loss_fake, self.d_loss,
                     self.g_gan_loss, self.image_loss, self.feature_loss, self.g_loss],
                     feed_dict={ self.source_images:source_batch, self.target_images:target_batch })
                
                
                counter += 1
                print('Epoch: %2d [%4d/%4d] time: %4.4f' % (epoch, idx, batch_idxs, time.time() - start_time))
                print('    d_loss： %.6f, d_gan_loss_real: %.6f, d_gan_loss_fake: %.6f' % (d_loss, d_gan_loss_real, d_gan_loss_fake))
                print('    g_loss: %.6f, g_gan_loss: %.6f, image_loss: %.6f, feature_loss: %.6f' % (g_loss, g_gan_loss, image_loss, feature_loss))

                if np.mod(counter, 10) == 1:
                        samples = self.sess.run([self.sampler],feed_dict={self.sample_images: sample_soruce_image})
                        #inverse
                        samples = np.squeeze(samples, axis==3)
                        samples = (samples * 128.) + 128
                        samples.dtype = 'uint8'
                        render_frame(samples, config.frame_dir, counter)
                        
                if np.mod(counter, 20) == 1:        
                        summary_str = self.sess.run(self.summary_all, feed_dict={ self.source_images:source_batch, self.target_images:target_batch })
                        self.summary_writer.add_summary(summary_str, counter)                           

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)      

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False    
        
        
        