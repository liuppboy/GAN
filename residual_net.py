# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def dense_layer(x, input_dim, output_dim, stddev=0.01, activation=None, name='dense_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
        Wx_b = tf.matmul(x, W) + b
        if activation:
            Wx_b = activation(Wx_b)
    return Wx_b

def batch_norm(x, is_train, is_reuse=None, decay=0.9, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        bn = tf.contrib.layers.batch_norm(x, decay=decay, center=True, scale=True, epsilon=1e-5, 
                is_training=is_train, updates_collections=None, reuse=is_reuse, scope=scope)
    return bn

#def batch_norm(x, is_train, name='batch_norm'):
    #"""
    #Batch normalization on convolutional maps.
    #Modified from: https://goo.gl/ckZxs8
    #answered by user http://stackoverflow.com/users/3632556/bgshi
    #"""
    #with tf.variable_scope(name):
        #output_channels = x.get_shape()[-1]
        #beta = tf.get_variable('beta', shape=[output_channels], initializer=tf.constant_initializer(0.0))
        #gamma = tf.get_variable('gamma', shape=[output_channels], initializer=tf.constant_initializer(1.0))
        #batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        #ema = tf.train.ExponentialMovingAverage(decay=0.9)

        #def mean_var_with_update():
            #ema_apply_op = ema.apply([batch_mean, batch_var])
            #with tf.control_dependencies([ema_apply_op]):
                #return tf.identity(batch_mean), tf.identity(batch_var)
        
        #if is_train:
            #mean, var = mean_var_with_update()
        #else:
            #mean, var = ema.average(batch_mean), ema.average(batch_var)
        ##mean, var = tf.cond(is_train,
                            ##mean_var_with_update,
                            ##lambda: (ema.average(batch_mean), ema.average(batch_var)))
        #normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    #return normed

def conv2d(x, filter_shape, strides=[1, 1, 1, 1], stddev=0.01, padding='SAME', name='conv2d'):
    with tf.variable_scope(name):
        W = tf.get_variable('W', filter_shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', filter_shape[-1], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(x, W, strides, padding)
        conv = tf.nn.bias_add(conv, b)
    return conv

def conv2d_block(x, is_train, filter_shape, strides=[1, 1, 1, 1], stddev=0.01, 
                 activation=tf.nn.relu, padding='SAME', name='conv2d_block'):
    with tf.variable_scope(name):
        conv = conv2d(x, filter_shape, strides, stddev, padding)
        bn = batch_norm(conv, is_train)
        acti = activation(bn)
    return acti
        

def conv2d_transpose(x, filter_shape, strides=[1, 1, 1, 1], stddev = 0.01, 
                     padding='SAME', name='conv2d_transpose'):
    with tf.variable_scope(name):        
        # filter : [height, width, output_channels, in_channels]
        output_channels = filter_shape[-2]
        W = tf.get_variable('W', filter_shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', output_channels, initializer=tf.constant_initializer(0.1))
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, output_channels]
        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding='SAME')
        deconv = tf.nn.bias_add(deconv, b)
    return deconv

def conv2d_transpose_block(x, is_train, filter_shape, strides=[1, 1, 1, 1], stddev=0.01, 
                           activation=tf.nn.relu, padding='SAME', name='conv2d_transpose_block'):
    with tf.variable_scope(name):
        conv = conv2d_transpose(x, filter_shape, strides, stddev, padding)
        bn = batch_norm(conv, is_train)
        acti = activation(bn)
    return acti    


#def residual_net(x, output_channels, activation=tf.nn.relu, is_train=tf.constant(True), 
                 #is_downsample=False, is_projection=False, name='resnet'):
    ## assume kernel size = 3, dowmsaple stride = 2
    #with tf.variable_scope(name):
        #input_channels = x.get_shape()[-1]
        #resnet = x
        
        #resnet = batch_norm(resnet, is_train)
        #resnet = activation(resnet)
        #if is_downsample:
            #resnet = conv2d(resnet, filter_shape=[1, 3, 3, 1], strides=[1, 2, 2, 1])
        #else:
            #resnet = conv2d(resnet, filter_shape=[1, 3, 3, 1])
        
        #resnet = batch_norm(resnet, is_train)
        #resnet = activation(resnet)
        #resnet = conv2d(resnet, filter_shape=[1, 3, 3, 1])
    
        ##downsample
        #if is_downsample:
            #x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #if input_channels != output_channels:
            #if is_projection:
                ## Option B: Projection shortcut
                #x = conv2d(x, [1, 1, input_channels, output_channels], strides=[1, 2, 2, 1])                
            #else:
                ## Option A: Zero-padding
                #ch = (output_channels - input_channels)//2
                #x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [ch, ch]])
        #resnet += x
        
    #return resnet

def residual_net(x, output_channels, is_train, activation=tf.nn.relu, 
                 is_downsample=False, is_projection=False, name='resnet'):
    # assume kernel size = 3, dowmsaple stride = 2
    with tf.variable_scope(name):
        input_channels = x.get_shape().as_list()[-1]
        resnet = x
        
        if is_downsample:
            resnet = conv2d(resnet, filter_shape=[3, 3, input_channels, output_channels], strides=[1, 2, 2, 1], name='conv2d_1')
        else:
            resnet = conv2d(resnet, filter_shape=[3, 3, input_channels, output_channels], name='conv2d_1')       
        resnet = batch_norm(resnet, is_train, name='batch_norm_1')
        resnet = activation(resnet)
        
        resnet = conv2d(resnet, filter_shape=[3, 3, output_channels, output_channels], name='conv2d_2')
        resnet = batch_norm(resnet, is_train, name='batch_norm_2')
        resnet = activation(resnet)
    
        #downsample
        if is_downsample:
            x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        if input_channels != output_channels:
            if is_projection:
                # Option B: Projection shortcut
                x = conv2d(x, [1, 1, input_channels, output_channels], strides=[1, 2, 2, 1], name='projection')                
            else:
                # Option A: Zero-padding
                ch = (output_channels - input_channels)//2
                x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [ch, ch]])
        resnet += x
        
    return resnet
def residual_block(x, n_blocks, output_channels, is_train, activation=tf.nn.relu, 
                   is_downsample=False, is_projection=False, name='resnet_block'):
    with tf.variable_scope(name):
        if n_blocks > 0:
            resnet = x
            residual_name = 'residual_%d' % 0
            resnet = residual_net(resnet, output_channels, is_train, activation, is_downsample, is_projection, name=residual_name)
            for i in range(1, n_blocks):
                residual_name = 'residual_%d' % i
                resnet = residual_net(resnet, output_channels, is_train, activation, False, is_projection, name=residual_name)
    
    return resnet