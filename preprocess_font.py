# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import scipy.misc as misc


def render_fonts_image(x, path, img_per_row, unit_scale=True):
    if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
        bitmaps = (x * 255.).astype(dtype=np.int16) % 256
    else:
        bitmaps = x
    num_imgs, w, h = x.shape
    assert w == h
    side = int(w)
    width = img_per_row * side
    height = int(np.ceil(float(num_imgs) / img_per_row)) * side
    canvas = np.zeros(shape=(height, width), dtype=np.int16)
    # make the canvas all white
    canvas.fill(255)
    for idx, bm in enumerate(bitmaps):
        x = side * int(idx / img_per_row)
        y = side * int(idx % img_per_row)
        canvas[x: x + side, y: y + side] = bm
    misc.toimage(canvas).save(path)
    return path

FLAGS = None

def draw_char_bitmap(ch, font, font_size, x_offset, y_offset):
    image = Image.new("RGB", (font_size, font_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    gray = image.convert('L')
    bitmap = np.asarray(gray)
    return bitmap

def generate_font_bitmaps(chars, font_path, font_size, canvas_size, x_offset, y_offset):
    font_obj = ImageFont.truetype(font_path, font_size)
    bitmaps = list()
    for c in chars:
        bm = draw_char_bitmap(c, font_obj, canvas_size, x_offset, y_offset)
        bitmaps.append(bm)
    return np.array(bitmaps)


def process_font(chars, font_path, save_dir, canvas_size, font_size):
    x_offset = (canvas_size - font_size) / 2
    y_offset = x_offset
    font_bitmaps = generate_font_bitmaps(chars, font_path, font_size,
                                         canvas_size, x_offset, y_offset)
    _, ext = os.path.splitext(font_path)
    if not ext.lower() in [".otf", ".ttf"]:
        raise RuntimeError("unknown font type found %s. only TrueType or OpenType is supported" % ext)
    _, tail = os.path.split(font_path)
    font_name = ".".join(tail.split(".")[:-1])
    bitmap_path = os.path.join(save_dir, "%s.npy" % font_name)
    np.save(bitmap_path, font_bitmaps)
    sample_image_path = os.path.join(save_dir, "%s_sample.png" % font_name)
    render_fonts_image(font_bitmaps[:100], sample_image_path, 10, False)
    print("font %s saved at %s" % (font_name, bitmap_path))


def get_chars_set(path):
    """
    Expect a text file that each line is a char
    """
    chars = list()
    try:
        with open(path, encoding="utf8") as f:
            for line in f:
                line = u"%s" % line
                char = line.split()[0]
                chars.append(char)            
    except:
        reload(sys)
        sys.setdefaultencoding("utf-8")    
        with open(path) as f:  
            for line in f:
                line = u"%s" % line
                char = line.split()[0]
                chars.append(char)            
    
    return chars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--canvas_size', type=int, default=64,
                        help='total image size')
    parser.add_argument('--font_size', type=int, default=48,
                        help='font size in image')
    FLAGS = parser.parse_args()

    save_dir = 'font_img'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
       
    char_list = 'charsets/top_3000_simplified.txt'
    chars = get_chars_set(char_list)
    source_ttf = 'font/SIMSUN.ttf'
    target_font = 'font/MSYaHei.ttf'
    
    if source_ttf:
        process_font(chars, source_ttf, save_dir, FLAGS.canvas_size, FLAGS.font_size)
    if target_font:
        process_font(chars, target_font, save_dir, FLAGS.canvas_size, FLAGS.font_size)    
        