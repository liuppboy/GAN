# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import xrange

import tensorflow as tf
import numpy as np

from residual_net import dense_layer, residual_net, residual_block, batch_norm, conv2d, conv2d_block, conv2d_transpose, conv2d_transpose_block


class GAN(object):
    def __init__(self, sess, batch_size=64, image_size=64, input_channels=1):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_channels = input_channels
        
    def encoder(self, x, is_train, is_reuse=False):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()            
            e_conv_1 = conv2d_block(x, is_train, filter_shape=[5, 5, self.input_channels, 64], name='e_conv_1')
            e_residual_block_1 = residual_block(e_conv_1, 3, 128, is_train=is_train, is_downsample=True, name='e_resblock_1')
            e_residual_block_2 = residual_block(e_residual_block_1, 3, 256, is_train=is_train, is_downsample=True, name='e_resblock_2')
            e_residual_block_3 = residual_block(e_residual_block_1, 3, 512, is_train=is_train, is_downsample=True, name='e_resblock_3')
        return e_residual_block_3
    
    def generator(self, x, is_train, is_reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()            
            g_conv_1 = conv2d_block(x, is_train, filter_shape=[5, 5, x.get_shape[0], 512], name='g_conv2d_1')
            g_deconv_1 = conv2d_transpose_block(g_conv_1, is_train, filter_shape=[5, 5, 256, 512], name='g_deconv_1')
            g_residual_block_1 = residual_block(g_deconv_1, 3, 256, is_train, is_downsample=False, name='g_resblock_1')
            g_deconv_2 = conv2d_transpose_block(g_residual_block_1, is_train, filter_shape=[5, 5, 128, 256], name='g_deconv_2')
            g_residual_block_2 = residual_block(g_deconv_1, 3, 128, is_train, is_downsample=False, name='g_resblock_2')
            g_deconv_3 = conv2d_transpose_block(g_residual_block_3, is_train, filter_shape=[5, 5, 64, 128], name='g_deconv_3')
            g_conv_2 = conv2d_block(g_deconv_3, is_train, filter_shape=[5, 5, 64, self.input_channels], activation=tf.nn.tanh, name='g_conv_2')
        return g_conv_2
    
    def discriminator(self, x, is_train, is_reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()            
            d_conv_1 = conv2d_block(x, is_train, filter_shape=[5, 5, x.get_shape[0], 64], name='d_conv2d_1')
            d_residual_block_1 = residual_block(e_conv_1, 4, 128, is_train=is_train, is_downsample=True, name='d_resblock_1')
            d_residual_block_2 = residual_block(e_residual_block_1, 4, 256, is_train=is_train, is_downsample=True, name='d_resblock_2')
            d_residual_block_3 = residual_block(e_residual_block_1, 4, 512, is_train=is_train, is_downsample=True, name='d_resblock_3')
            d_conv_2 = conv2d_block(d_residual_block_3, is_train, filter_shape=[5, 5, x.get_shape[0], 512], name='d_conv2d_2')
            d_reshape = tf.reshape(d_conv_2, [self.batch_size, -1], name='d_reshape')
            d_output = dense_layer(d_reshape, d_reshape.get_shape()[-1], 1, activation=tf.nn.sigmoid, name='d_dense_layer')
        return d_output
    
    def bulid_model(self):  
        
        
        
        
        
        