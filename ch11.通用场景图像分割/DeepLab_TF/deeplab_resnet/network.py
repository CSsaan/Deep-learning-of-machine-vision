import numpy as np
from six import string_types
import tensorflow as tf

DEFAULT_PADDING = 'SAME'

def layer(tf_op):
    ''' 
    Python装饰器的用法，具体教程看这个链接：
    http://wiki.jikexueyuan.com/project/explore-python/Functional/decorator.html
    装饰器的作用是：封装成可以组装的基本网络层（卷积、膨胀卷积、池化等），方便组装复杂网络
    '''
    def layer_decorated(self, *args, **kwargs):
        '''
        装饰器用于装饰(包裹、封装)原有函数的输出，返回的是包装后的函数layer_decorated
        需要装饰的函数，在其函数名上方追加装饰器@layer
        '''
        # 如果没有提供‘name’参数，自动根据操作名称提供一个unique名称
        name = kwargs.setdefault('name', self.get_unique_name(tf_op.__name__))
        # 提取操作层的输入
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # 调用原函数，获得操作层的输出
        layer_output = tf_op(self, layer_input, *args, **kwargs)
        # 将输出存入类中的self.layers变量
        self.layers[name] = layer_output
        # 把当前输出缓存为下一层的输入
        self.feed(layer_output)
        # 返回类对象本身，以实现链式调用
        return self

    # 返回包装函数layer_decorated
    return layer_decorated



class Network(object):
    ''' 
    基础网络类，提供基本操作
    '''
    def __init__(self, inputs, trainable=True, is_training=False, num_classes=21):
        # 网络的输入节点
        self.inputs = inputs
        # 当前操作层的输出终端节点list，下一个操作层的输入缓存
        self.terminals = []
        # 操作层缓存器，存储网络输入节点和各层的输出节点
        self.layers = dict(inputs)
        # 是否训练
        self.trainable = trainable
        # dropout的开关变量
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        # 开始构建网络
        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''网络构建函数，需要子类做实现 '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''加载网络权重参数
        data_path: numpy-serialized网络权重参数的文件路径
        session: 当前的 TensorFlow session
        ignore_missing: 是否忽略丢失层的serialized权重参数
        '''
        # 读取numpy-serialized网络权重参数的文件
        data_dict = np.load(data_path).item()
        # 遍历scope
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                # 遍历权重参数名称和权重参数值
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        # 获取权重参数变量
                        var = tf.get_variable(param_name)
                        # 赋值权重参数变量
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        # 清空操作层的输入缓存
        self.terminals = []
        # 将函数输入存入 输入缓存
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            # 输入缓存
            self.terminals.append(fed_layer)
        # 返回类对象
        return self

    def get_output(self):
        '''获取当前操作时刻的网络输出节点'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''基于prefix输入名称，生成unique名称
        '''
        # 为prefix输入名称（例如 conv），计算索引值（例如 1）
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        # 在prefix输入名称后，追加索引值名称（例如 conv_1）
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''新建新的TensorFlow变量.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''验证边界补值类型：补零，不补值'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        '''2D卷积'''
        
        # 验证padding类型
        self.validate_padding(padding)
        # 获取输入通道数量
        c_i = input.get_shape().as_list()[-1]
        # 验证group参数
        assert c_i % group == 0
        assert c_o % group == 0
        # 基于不同的输入和卷积核，定义卷积函数
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            # 为卷积核，新建权重参数
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # 普通卷积
                output = convolve(input, kernel)
            else:
                # 拆分通道，进行分组卷积
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # 串接分组卷积结果
                output = tf.concat(3, output_groups)
            # 添加篇置
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU 激活函数
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        '''2D膨胀卷积'''
        
        # 验证padding类型
        self.validate_padding(padding)
        # 获取输入通道数量
        c_i = input.get_shape().as_list()[-1]
        # 验证group参数
        assert c_i % group == 0
        assert c_o % group == 0
        # 基于不同的输入和卷积核，定义膨胀卷积函数
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            # 为卷积核，新建权重参数
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # 普通卷积
                output = convolve(input, kernel)
            else:
                # 拆分通道，进行分组膨胀卷积
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # 串接分组膨胀卷积结果
                output = tf.concat(3, output_groups)
            # 添加篇置
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU 激活函数
                output = tf.nn.relu(output, name=scope.name)
            return output
        
    @layer
    def relu(self, input, name):
        '''ReLU激活'''
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        '''最大池化'''
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        '''均值池化'''
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        '''局部响应归一化'''
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        '''通道串接'''
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        '''逐像素相加'''
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        '''全连接层'''
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        '''softmax层'''
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)
        
    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
        '''BN层'''
        with tf.variable_scope(name) as scope:
            output = tf.contrib.layers.batch_norm(
                input,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        '''dropout层'''
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
