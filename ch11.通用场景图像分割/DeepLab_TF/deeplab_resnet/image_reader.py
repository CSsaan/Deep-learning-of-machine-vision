import os

import numpy as np
import tensorflow as tf
from .utils import label_colours

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
   
    return img, label

def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop  

def read_labeled_image_list(data_dir, data_list):
    """读取txt文件，获取图片和标注标注图片的相对路径列表.
    
    Args:
      data_dir: 数据集路径.
      data_list: 相对路径列表txt文件，格式： '/path/to/image /path/to/mask'.
       
    Returns:
      两个文件路径列表.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    """读取图片和标注图片内容.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    
    # 批量读取图片和标注图片
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    # 解析图片，变换通道顺序：RGB-->BGR
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    
    # 减去图片均值
    img -= img_mean

    # 解析标注图片
    label = tf.image.decode_png(label_contents, channels=1)
    
    # 随机变换
    if input_size is not None:
        h, w = input_size

        # 随机尺寸变换images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # 随机镜像images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # 随机剪裁images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, 
                 random_scale, random_mirror, ignore_label, img_mean, coord):
        '''图片读取器ImageReader.
        
        Args:
          data_dir: 数据集路径.
          data_list: 相对路径列表txt文件，格式： '/path/to/image /path/to/mask'.
          input_size: 输入h尺寸 (height, width) 
          random_scale: 是否随机变换尺寸.
          random_mirror: 是否随机变换镜像.
          ignore_label: 忽略的label值.
          img_mean: 图片像素均值.
          coord: 队列协调器TensorFlow queue coordinator.
        '''
        # 初始化赋值
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        
        # 读取图片和标注图片的路径列表
        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, 
                                                                   self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        
        # 构建随机队列
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val

        # 根据路径队列读取图片和标注内容
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, 
                                                       random_scale, random_mirror, 
                                                       ignore_label, img_mean) 

    def dequeue(self, num_elements):
        '''
        生成batch训练数据：images、labels
        Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          2个尺寸为 (batch_size, h, w, {3, 1}) 的tensor: images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch
