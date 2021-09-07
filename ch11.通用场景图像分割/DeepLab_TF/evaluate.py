from __future__ import print_function
import argparse
import tensorflow as tf
import numpy as np
from deeplab_resnet import DeepLabResNetModel, ImageReader

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './dataset'
DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
RESTORE_FROM = './ini_model/model.ckpt-40000'

def get_arguments():
    """解析控制台参数.
    
    Returns:参数列表
    """
    parser = argparse.ArgumentParser(description="DeepLab Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
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
    
    # 构建队列协调器queue coordinator
    coord = tf.train.Coordinator()
    
    # 加载读取器reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # 无输入尺寸设置.
            False, # 无随机尺寸变换.
            False, # 无镜像变换.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    # 添加Batch维度.
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) 

    # 构建DeepLab-ResNet-101网络
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # 设定要预加载的网络权重参数
    restore_var = tf.global_variables()
    
    # 执行预测.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # 计算mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    
    # 建立tf session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # 执行权重变量初始化
    init = tf.global_variables_initializer()    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # 加载已有的checkpoint文件
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # 开启队列执行器线程.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # 遍历所有step.
    for step in range(args.num_steps):
        preds, _ = sess.run([pred, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    
    # 停止训练协调器
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
