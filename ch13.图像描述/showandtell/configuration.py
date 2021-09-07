# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    # sharded TFRecord文件的命名模式
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    # 每个TFRecord文件的大约数量量
    self.values_per_input_shard = 2300
    # Minimum number of shards to keep in the input queue.
    # 输入队列的最少shards数量
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    # 读线程数量
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    # 包含图片数据的SequenceExample context feature名称
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer captions.
    # 包含caption word_id数据的SequenceExample feature list名称
    self.caption_feature_name = "image/caption_ids"

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    # 字典尺寸
    self.vocab_size = 12000

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Batch size.
    self.batch_size = 32

    # File containing an Inception v3 checkpoint to initialize the variables
    # of the Inception model. Must be provided when starting training for the
    # first time.
    # Inception v3的pre-trained模型文件，首次训练需要提供
    self.inception_checkpoint_file = None

    # Dimensions of Inception v3 input images.
    # Inception v3的图片输出尺寸
    self.image_height = 299
    self.image_width = 299

    # Scale used to initialize model variables.
    # 模型变量初始化Scale
    self.initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively.
    # LSTM的输入、输出维度
    self.embedding_size = 512
    self.num_lstm_units = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    # lstm随机失活
    self.lstm_dropout_keep_prob = 0.7


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    # 每个epoch的数量量
    self.num_examples_per_epoch = 586363

    # Optimizer for training the model.
    # 优化方法
    self.optimizer = "SGD"

    # Learning rate for the initial phase of training.
    # 学习率
    self.initial_learning_rate = 2.0
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 8.0

    # Learning rate when fine tuning the Inception v3 parameters.
    # 调优Inception v3 模型参数的学习率
    self.train_inception_learning_rate = 0.0005

    # If not None, clip gradients to this value.
    # 梯度剪裁
    self.clip_gradients = 5.0

    # How many model checkpoints to keep.
    # 可保留的最大checkpoints模型文件数量
    self.max_checkpoints_to_keep = 5
