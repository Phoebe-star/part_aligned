
import matplotlib.pyplot as plt
import tensorflow as tf


import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 
image_raw_data = tf.gfile.FastGFile("data_eye/labeled/train/0000_08.jpg", 'rb').read()

with tf.Session() as sess:
    # 
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 
    resized = tf.image.resize_images(img_data, [224, 224], method=0)
    plt.imshow(resized.eval())
    #plt.show()

    # 
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 224, 224)
    plt.imshow(croped.eval())
    #plt.show()

    padded = tf.image.resize_image_with_crop_or_pad(img_data, 224, 224)
    plt.imshow(padded.eval())
    plt.show()

    # 
    #central_cropped = tf.image.random_brightness(img_data, 0.4)
    
    
    
    central_cropped = tf.image.central_crop(resized,random.uniform(0.7, 1))
    print central_cropped
    plt.imshow(central_cropped.eval())
    plt.show()
    


#https://liqiang311.github.io/tensorflow/TensorFlow-%E5%9B%BE%E5%83%8F%E9%A2%84%E5%A4%84%E7%90%86%E5%B8%B8%E7%94%A8%E6%89%8B%E6%AE%B5/