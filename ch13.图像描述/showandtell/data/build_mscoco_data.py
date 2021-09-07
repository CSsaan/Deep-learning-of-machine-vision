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
"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...

and

  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:

  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256

and

  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004

and

  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008

Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace

  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.

NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.

Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading



import nltk.tokenize
import numpy as np
import tensorflow as tf

# 图片文件路径
tf.flags.DEFINE_string("train_image_dir", "mscoco/raw-data/train2014/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "mscoco/raw-data/val2014",
                       "Validation image directory.")

# 标注文件路径
tf.flags.DEFINE_string("train_captions_file", "mscoco/raw-data/annotations/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "mscoco/raw-data/annotations/captions_val2014.json",
                       "Validation captions JSON file.")
# 输出路径
tf.flags.DEFINE_string("output_dir", "mscoco/tfrecord-data/", "Output data directory.")

# TFRecord文件中的shards数量
tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

# caption描述标注的设置：起始标记、结束标记、未知标记
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")

# 词汇字典的设置：存入字典的出现频次、字典文件路径
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "mscoco/raw-data/word_counts.txt",
                       "Output vocabulary file of word counts.")

# 图片处理线程
tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

# 图片集Metadata类型定义
ImageMetadata = namedtuple("ImageMetadata", ["image_id", "filename", "captions"])


class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # 为所有图片解码调用，建立单个.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    # 定义JPEG解码操作函数
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    # 在self._sess上执行解码操作
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _int64_feature(value):
  """ 封装函数：插入int64 Feature到SequenceExample proto
  Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """ 封装函数：插入bytes Feature到SequenceExample proto
  Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _str_bytes_feature(value):
  """ 封装函数：插入bytes Feature到SequenceExample proto
  Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _int64_feature_list(values):
  """封装函数：插入int64 FeatureList到SequenceExample proto
  Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """封装函数：插入bytes FeatureList到SequenceExample proto
  Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_str_bytes_feature(v) for v in values])


def _to_sequence_example(image, decoder, vocab):
  """为单个image-caption建立SequenceExample proto.

  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.

  Returns:
    A SequenceExample proto.
  """
  # 读取图片文件原始数据
  # print(image.filename)
  with tf.gfile.FastGFile(image.filename, "rb") as f:
    encoded_image = f.read()

  # 尝试解码为图片内容数据
  try:
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return

  # 为tf.train.SequenceExample构建context
  context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
  })

  # 为tf.train.SequenceExample构建feature_lists
  assert len(image.captions) == 1
  caption = image.captions[0]
  caption_ids = [vocab.word_to_id(word) for word in caption]
  feature_lists = tf.train.FeatureLists(feature_list={
      #"image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
  })
    
  # 建立tf.train.SequenceExample实例
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
  """单线程操作：处理并存储images子集为TFRecord文件.

  Args:
    thread_index: 线程索引/Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: 数据集名称/Unique identifier specifying the dataset.
    images: ImageMetadata数据列表/List of ImageMetadata.
    decoder: 图片解码器/An ImageDecoder object.
    vocab: 词汇字典/A Vocabulary object.
    num_shards: 单个文件的数据shards数量/Number of shards for the output files.
  """
  # 每个线程处理N shards， N = num_shards / num_threads. 
  # 例如，如果num_shards = 256 和 num_threads = 8,那么第一个线程处理shards [0, 32).
  # 整个数据集拆分逻辑：
  # 1）整个数据集分为256个shard组，并存储为256个TFRecord文件，
  # 2）每个线程依次负责存储32个shard组，并存为TFRecord文件
  # 3) 相应文件名例如：'train-00001-of-00256'
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  # 遍历每个shard组
  counter = 0
  for s in range(num_shards_per_batch):
    # 拼接Shard文件路径
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    # 建立TFRecordWriter写文件器
    writer = tf.python_io.TFRecordWriter(output_file)

    # 遍历每个shard组中的image-caption对
    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in images_in_shard:
      # 获取单个image-caption对
      image = images[i]

      # 转换为SequenceExample proto（二进制数据）
      sequence_example = _to_sequence_example(image, decoder, vocab)
      # 序列化，并写入TFRecord文件
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      # 打印log
      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()

    # 关闭写文件器
    writer.close()
    print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  """将数据集存储为TFRecord文件.

  Args:
    name: 数据集名称 / Unique identifier specifying the dataset.
    images: ImageMetadata数据集列表 / List of ImageMetadata.
    vocab: 词汇字典 / A Vocabulary object.
    num_shards: 单个文件的数据shards数量 / Number of shards for the output files.
  """
  # Break up each image into a separate entity for each caption.
  # 因为一个图片有5个caption，所以需要为每个caption存储一个图片，形成ImageMetadata列表
  images = [ImageMetadata(image.image_id, image.filename, [caption])
            for image in images for caption in image.captions]

  # 随机打乱images中的元素存储顺序
  random.seed(12345)
  random.shuffle(images)

  # 将images分成num_threads个batches. 
  # Batch i定义为images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # 建立协调器，监控所有线程threads是否结束.
  coord = tf.train.Coordinator()

  # 建立JPEG图片解码器
  decoder = ImageDecoder()

  # 为每个batch启动一个thread .
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # 等待所有线程threads终止
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _create_vocab(captions):
  """建立词汇字典/vocabulary of word to word_id.
  文件名word_counts.txt,共包含3个信息：
  1）词: 第一列
  2）词频：第二列
  3）词id: 0-based行号

  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.

  Args:
    captions: 所有caption字符串 / A list of lists of strings.

  Returns:
    字典对象.
  """
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # 滤除低频词汇，按词频降续排列.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # 生成字典文件.
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # 生成词汇字典.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


def _process_caption(caption):
  """把caption字符串转换成符号化的词汇列表(a list of tonenized words).

  Args:
    caption: caption字符串.

  Returns:
    词汇字符串列表; the tokenized caption.
  """
  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(FLAGS.end_word)
  return tokenized_caption


def _load_and_process_metadata(captions_file, image_dir):
  """从JSON文件中读取图片的metadata数据，处理captions标注集.

  Args:
    captions_file: 含有caption标注集的JSON文件.
    image_dir: 图片文件路径.

  Returns:
    ImageMetadata数据列表.
  """
  
  # 读取json文件
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)

  # 提取id和文件名.
  id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

  # 提取captions标注. 每个图片id对应多个captions标注.
  id_to_captions = {}
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])
    id_to_captions[image_id].append(caption)

  # 数量检查
  assert len(id_to_filename) == len(id_to_captions)
  assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
  print("Loaded caption metadata for %d images from %s" %
        (len(id_to_filename), captions_file))

  # 处理captions标注集，并生成ImageMetadata数据列表.
  print("Processing captions.")
  image_metadata = []
  num_captions = 0
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(image_dir, base_filename)
    captions = [_process_caption(c) for c in id_to_captions[image_id]]
    image_metadata.append(ImageMetadata(image_id, filename, captions))
    num_captions += len(captions)
  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))

  return image_metadata


def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """返回 True， 如果num_shards跟处理线程数量FLAGS.num_threads兼容"""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  # 检查train_shards、val_shards、test_shards跟FLAGS.num_threads的兼容性
  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

  # 检查输出路径，没有则新建
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # 从caption标注集文件中，加载image metadata.
  mscoco_train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,
                                                    FLAGS.train_image_dir)
  mscoco_val_dataset = _load_and_process_metadata(FLAGS.val_captions_file,
                                                  FLAGS.val_image_dir)

  # MSCOCO data数据集再分配:
  # 新train集：全部的mscoco_train_dataset + 前85%的mscoco_val_dataset
  #   train_dataset = 100% of mscoco_train_dataset + 85% of mscoco_val_dataset.
  # 新val集：mscoco_val_dataset中接下来的5%
  #   val_dataset = 5% of mscoco_val_dataset (for validation during training).
  # 新test集：mscoco_val_dataset中剩余的10%
  #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).
  train_cutoff = int(0.85 * len(mscoco_val_dataset))
  val_cutoff = int(0.90 * len(mscoco_val_dataset))
  train_dataset = mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
  val_dataset = mscoco_val_dataset[train_cutoff:val_cutoff]
  test_dataset = mscoco_val_dataset[val_cutoff:]

  # 基于training captions建立词汇字典.
  train_captions = [c for image in train_dataset for c in image.captions]
  vocab = _create_vocab(train_captions)

  # 开始处理
  _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
  _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
  _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
  tf.app.run()
