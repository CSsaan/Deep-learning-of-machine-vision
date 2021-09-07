# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# BN层由slim库提供
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from .network import Network
import tensorflow as tf

class DeepLabResNetModel(Network):
    def setup(self, is_training_bn, num_classes):
        '''DeepLab-ResNet-101网络定义，基于ResNet-101.
        
        Args:
          is_training: 是否更新BN层（在测试阶段）的均值和标准差.
                      如果batch size小，建议不更新，直接使用pre-trained模型（测试阶段）的均值和标准差。
          num_classes: 需要预测的类别数量（包含背景类）.
        '''
        # 第一卷基层，无残差直连
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
        # 残差直连的通道升维，因为此处特征的2D尺寸有降维
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn2a_branch1'))

        # 第一卷积组，第1残差单元
        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn2a_branch2c'))

        # 第二卷积组，第2残差单元
        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn2b_branch2c'))

        # 第二卷积组，第3残差单元
        (self.feed('res2a_relu', 
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn2c_branch2c'))
        
        # 第三卷积组
        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
        # 残差直连的通道升维，因为此处特征的2D尺寸有降维
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn3a_branch1'))

        # 第三卷积组，第1残差单元
        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn3a_branch2c'))

        # 第三卷积组，第2残差单元
        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn3b1_branch2c'))

        (self.feed('res3a_relu', 
                   'bn3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn3b2_branch2c'))

        # 第三卷积组，第3残差单元
        (self.feed('res3b1_relu', 
                   'bn3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn3b3_branch2c'))

        # 第四卷积组
        (self.feed('res3b2_relu', 
                   'bn3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu')
        # 残差直连的通道升维，因为此处特征的2D尺寸有降维
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4a_branch1'))

        # 第四卷积组，第1残差单元
        (self.feed('res3b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4a_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4a_branch2c'))

        # 第四卷积组，第2残差单元
        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b1_branch2c'))

        # 第四卷积组，第3残差单元
        (self.feed('res4a_relu', 
                   'bn4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b2_branch2c'))

        # 第四卷积组，第4残差单元
        (self.feed('res4b1_relu', 
                   'bn4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b3_branch2c'))

        # 第四卷积组，第5残差单元
        (self.feed('res4b2_relu', 
                   'bn4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b4_branch2c'))

        # 第四卷积组，第6残差单元
        (self.feed('res4b3_relu', 
                   'bn4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b5_branch2c'))

        # 第四卷积组，第7残差单元
        (self.feed('res4b4_relu', 
                   'bn4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b6_branch2c'))

        # 第四卷积组，第8残差单元
        (self.feed('res4b5_relu', 
                   'bn4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b7_branch2c'))

        # 第四卷积组，第9残差单元
        (self.feed('res4b6_relu', 
                   'bn4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b8_branch2c'))

        # 第四卷积组，第10残差单元
        (self.feed('res4b7_relu', 
                   'bn4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b9_branch2c'))

        # 第四卷积组，第11残差单元
        (self.feed('res4b8_relu', 
                   'bn4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b10_branch2c'))

        # 第四卷积组，第12残差单元
        (self.feed('res4b9_relu', 
                   'bn4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b11_branch2c'))

        # 第四卷积组，第13残差单元
        (self.feed('res4b10_relu', 
                   'bn4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b12_branch2c'))

        # 第四卷积组，第14残差单元
        (self.feed('res4b11_relu', 
                   'bn4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b13_branch2c'))

        # 第四卷积组，第15残差单元
        (self.feed('res4b12_relu', 
                   'bn4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b14_branch2c'))

        # 第四卷积组，第16残差单元
        (self.feed('res4b13_relu', 
                   'bn4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b15_branch2c'))

        # 第四卷积组，第17残差单元
        (self.feed('res4b14_relu', 
                   'bn4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b16_branch2c'))

        # 第四卷积组，第18残差单元
        (self.feed('res4b15_relu', 
                   'bn4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b17_branch2c'))

        # 第四卷积组，第19残差单元
        (self.feed('res4b16_relu', 
                   'bn4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b18_branch2c'))

        # 第四卷积组，第20残差单元
        (self.feed('res4b17_relu', 
                   'bn4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b19_branch2c'))

        # 第四卷积组，第21残差单元
        (self.feed('res4b18_relu', 
                   'bn4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b20_branch2c'))

        # 第四卷积组，第22残差单元
        (self.feed('res4b19_relu', 
                   'bn4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b21_branch2c'))

        # 第四卷积组，第23残差单元
        (self.feed('res4b20_relu', 
                   'bn4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
             # atrous膨胀卷积，Rate = 2
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn4b22_branch2c'))

        # 第五卷积组
        (self.feed('res4b21_relu', 
                   'bn4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn5a_branch1'))

        # 第五卷积组，第1残差单元
        (self.feed('res4b22_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn5a_branch2a')
             # atrous膨胀卷积，Rate = 4
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn5a_branch2c'))

        # 第五卷积组，第2残差单元
        (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn5b_branch2a')
             # atrous膨胀卷积，Rate = 4
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn5b_branch2c'))

        # 第五卷积组，第3残差单元
        (self.feed('res5a_relu', 
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(is_training=is_training_bn, activation_fn=tf.nn.relu, name='bn5c_branch2a')
             # atrous膨胀卷积，Rate = 4
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training_bn)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(is_training=is_training_bn, activation_fn=None, name='bn5c_branch2c'))

        # 第六卷积组，atrous空间金字塔池化（4个并行膨胀卷积）
        (self.feed('res5b_relu', 
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu')             
        # 第六卷积组，膨胀卷积，Rate = 6分支  
             .atrous_conv(3, 3, num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

        # 第六卷积组，膨胀卷积，Rate = 12分支 
        (self.feed('res5c_relu')
             .atrous_conv(3, 3, num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

        # 第六卷积组，膨胀卷积，Rate = 18分支  
        (self.feed('res5c_relu')
             .atrous_conv(3, 3, num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))
        
        # 第六卷积组，膨胀卷积，Rate = 24分支 
        (self.feed('res5c_relu')
             .atrous_conv(3, 3, num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

        # 第六卷积组，4分支合并
        (self.feed('fc1_voc12_c0', 
                   'fc1_voc12_c1', 
                   'fc1_voc12_c2', 
                   'fc1_voc12_c3')
             .add(name='fc1_voc12'))
