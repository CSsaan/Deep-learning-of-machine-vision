from __future__ import print_function
import argparse
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from deeplab_resnet import DeepLabResNetModel, decode_labels

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21
SAVE_DIR = './output/'
IMAGE_PATH = './dataset/JPEGImages/2007_000039.jpg'
RESTORE_FROM = './ini_model/model.ckpt-40000'

def get_arguments():
    """解析控制台参数.
    
    Returns:参数列表
    """
    
    parser = argparse.ArgumentParser(description="DeepLab Network Inference.")
    parser.add_argument("--img-path", type=str, default=IMAGE_PATH,
                        help="Path to the RGB image file.")
    parser.add_argument("--model-weights", type=str, default=RESTORE_FROM,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''加载已训练的权重参数.
    
    Args:
      saver: TensorFlow Saver 存储器对象.
      sess: TensorFlow session.
      ckpt_path: checkpoint权重参数文件路径.
    ''' 

    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """主函数：模型构建和evaluate."""
    args = get_arguments()
    
    # 读取图片.
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # 通道转换： RGB --> BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # 减去像素均值.
    img -= IMG_MEAN 
    
    # 构建DeepLab-ResNet-101网络.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)

    # 设定要预加载的网络权重参数
    restore_var = tf.global_variables()

    # 执行预测.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # 建立tf session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # 执行权重变量初始化
    init = tf.global_variables_initializer()    
    sess.run(init)
    
    # 加载已有的checkpoint文件
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)
    
    # 执行推断.
    preds = sess.run(pred)
    
    msk = decode_labels(preds, num_classes=args.num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    im.save(args.save_dir + 'mask.png')
    
    print('The output file has been saved to {}'.format(args.save_dir + 'mask.png'))

    
if __name__ == '__main__':
    main()
