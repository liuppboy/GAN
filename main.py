import os
import numpy as np


import tensorflow as tf
from model import GAN

class Config():
    def __init__(self, epoch=25, learning_rate=0.001, beta1=0.9, train_size=2000, validation_size=100,
                 checkpoint_dir='checkpoint_dir', frame_dir='frame_dir', source_font='font_img/SIMSUN.npy', target_font='font_img/MSYaHei.npy'):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.train_size = train_size
        self.validation_size = validation_size
        self.checkpoint_dir = checkpoint_dir
        self.frame_dir = frame_dir
        self.source_font = source_font
        self.target_font = target_font


def main(_):
    
    config = Config()
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.frame_dir):
        os.makedirs(config.frame_dir)

    with tf.Session() as sess:
        
        gan = GAN(sess)
        gan.train(config)


if __name__ == '__main__':
    tf.app.run()
