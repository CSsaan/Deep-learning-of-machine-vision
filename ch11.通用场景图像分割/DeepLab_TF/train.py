""" 语义分割DeepLab-ResNet-101网络的训练代码
    数据集augmented PASCAL VOC（大约10000张训练images，1500张测试）
"""

from __future__ import print_function
import argparse
import os
import time
import tensorflow as tf
import numpy as np
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 4
DATA_DIRECTORY = './dataset'
DATA_LIST_PATH = './dataset/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 9e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 40001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './ini_model/model.ckpt-40000'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005


def get_arguments():
    """解析控制台参数.
    
    Returns:参数列表
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''存储权重参数.
   
   Args:
     saver: TensorFlow Saver 存储器对象.
     sess: TensorFlow session.
     logdir: snapshots文件存储路径.
     step: 当前训练step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

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
    """主函数：模型构建和训练"""
    
    # 获取训练参数
    args = get_arguments()
    
    # 获取输出尺寸
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # 设置随机种子
    tf.set_random_seed(args.random_seed)
    
    # 构建队列协调器queue coordinator
    coord = tf.train.Coordinator()
    
    # 加载读取器reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # 构建DeepLab-ResNet-101网络
    net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)

    # 获取网络输出.
    raw_output = net.layers['fc1_voc12']

    # 获取不同类型的网络权重参数名
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # 学习率lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # 学习率lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # 学习率lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    
    # 根据ground truth类别标签，提取
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
                                                  
                                                  
    # 构建各个像素的损失函数：交叉熵softmax loss + 权重衰减L2 loss
    # 交叉熵softmax loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    # 权重衰减L2 loss
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    # loss合并
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    
    # 预测结果可视化
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # 添加图片总结
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)    
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.                               
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # 定义loss和优化参数
    # 定义学习率
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    # 定义优化器
    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    # 获取loss梯度
    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    # 定义梯度优化执行操作
    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    
    
    # 建立tf session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # 执行权重变量初始化
    init = tf.global_variables_initializer()    
    sess.run(init)
    
    # 获取训练存储器
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    
    # 加载已有的checkpoint文件
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # 开启队列执行器线程.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # 遍历所有训练step.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        
        # 执行训练，并存储训练结果
        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(saver, sess, args.snapshot_dir, step)
        # 仅执行训练，不存储训练结果
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        if step % 10 == 0:
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    
    # 停止训练协调器
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
