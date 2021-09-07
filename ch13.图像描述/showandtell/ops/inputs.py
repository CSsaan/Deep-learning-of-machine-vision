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

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def parse_sequence_example(serialized, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.
  解析tensorflow.SequenceExample为图片和caption

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image, caption


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.
  从磁盘读取字符串形式的数值，到输入队列

  
  

  Args:
    reader: TFRecord读取器实例/Instance of tf.ReaderBase.
    file_pattern: TFRecord文件名模式/Comma-separated list of file patterns 
					(e.g. /tmp/train_data-?????-of-00256).
    is_training: 是否是训练模式/Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: 每个TFRecord文件的数据量/Approximate number of values per shard.
	
    input_queue_capacity_factor: 
		输入队列要保证的最小数据量，values_per_shard的倍数，
		即values_per_shard * input_queue_capacity_factor。
		大的倍数可以让不同TFRecord文件的数据量更好的进行随机混合。
		Minimum number of values to keep in the queue in multiples of values_per_shard.
		In training the capacity of the queue is important because a larger queue
		means better mixing of training examples between shards. The minimum number of
		values kept in the queue is values_per_shard * input_queue_capacity_factor,
		where input_queue_memory factor should be chosen to trade-off better mixing
		with memory usage.
		
    num_reader_threads: 读线程数量/Number of reader threads to fill the queue.
    shard_queue_name: TFRecord文件队列名称Name for the shards filename queue.
    value_queue_name: 数值队列名称Name for the values input queue.

  Returns:
    字符串形式的数据队列
	A Queue containing prefetched string values.
  """
  # 根据文件名模式，获取TFRecord文件名列表
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    # 文件名队列生成器
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    # 队列最小数据量
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    # 队列数据容量
    capacity = min_queue_examples + 100 * batch_size
    # 随机数值队列
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    # 文件名队列生成器
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    # 队列数据容量
    capacity = values_per_shard + 3 * batch_size
    # 先入先出队列
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  # 多线程
  enqueue_ops = []
  for _ in range(num_reader_threads):
    # 读文件队列
    _, value = reader.read(filename_queue)
    # 数据入队操作
    enqueue_ops.append(values_queue.enqueue([value]))
  # 添加队列执行器
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  # 添加随机数据队列的尺寸标量的总结
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.
  把image-caption对 列表batch化

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.
  1）caption转换成多组：输入word序列（input sequence）-->目标输出word序列（target sequence）
                       其中，目标输出word序列是输入word序列右移1个位置形成的
  2）全部补充（pad)到最长序列
  3）生成对应的真word序列mask
  4) 生成batch数据

  Example:
    Actual captions in the batch ('-' denotes padded character补充字符):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: 图片数据batch --> A Tensor of shape [batch_size, height, width, channels].
    input_seqs: 输入序列batch --> An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: 目标序列batch --> An int32 Tensor of shape [batch_size, padded_length].
    mask: 真word序列batch --> An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  # 生成入队列表
  enqueue_list = []
  for image, caption in images_and_captions:
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

    input_seq = tf.slice(caption, [0], input_length)
    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    enqueue_list.append([image, input_seq, target_seq, indicator])

  # 生成batch数据的队列
  images, input_seqs, target_seqs, mask = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  # 添加总结
  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask
