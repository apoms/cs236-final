# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes made by ogroth, stefan.

"""Minimal data reader for GQN TFRecord datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datasets.common
from datasets.common import (
  DatasetInfo, Context, Query, TaskData, _DATASETS, _NUM_CHANNELS,
  _NUM_RAW_CAMERA_PARAMS, _MODES, _get_dataset_files)

import collections
import os
import cv2
import tensorflow as tf
nest = tf.contrib.framework.nest

tf.enable_eager_execution()

def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class GQNTFRecordDataset(tf.data.Dataset):
  """Minimal tf.data.Dataset based TFRecord dataset.

  You can use this class to load the datasets used to train Generative Query
  Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
  See README.md for a description of the datasets and an example of how to use
  the class.
  """

  def __init__(self, dataset, root, mode='train',
               custom_frame_size=None, num_threads=4, buffer_size=256,
               parse_batch_size=32):
    """Instantiates a DataReader object and sets up queues for data reading.

    Args:
      dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
          'rooms_free_camera_no_object_rotations',
          'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
          'shepard_metzler_7_parts'].
      context_size: integer, number of views to be used to assemble the context.
      root: string, path to the root folder of the data.
      mode: (optional) string, one of ['train', 'test'].
      custom_frame_size: (optional) integer, required size of the returned
          frames, defaults to None.
      num_threads: (optional) integer, number of threads used when reading and
          parsing records, defaults to 4.
      buffer_size: (optional) integer, capacity of the buffer into which
          records are read, defualts to 256.
      parse_batch_size: (optional) integer, number of records to parse at the
          same time, defaults to 32.

    Raises:
      ValueError: if the required version does not exist; if the required mode
         is not supported; if the requested context_size is bigger than the
         maximum supported for the given dataset version.
    """

    if dataset not in _DATASETS:
      raise ValueError('Unrecognized dataset {} requested. Available datasets '
                       'are {}'.format(dataset, _DATASETS.keys()))

    if mode not in _MODES:
      raise ValueError('Unsupported mode {} requested. Supported modes '
                       'are {}'.format(mode, _MODES))

    self._dataset_info = _DATASETS[dataset]

    # Number of views in the context + target view
    self._example_size = 1
    self._custom_frame_size = custom_frame_size

    self._feature_map = {
      'frames': tf.FixedLenFeature(
        shape=self._dataset_info.sequence_size, dtype=tf.string),
      'cameras': tf.FixedLenFeature(
        shape=[self._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
        dtype=tf.float32)
    }

    file_names = _get_dataset_files(self._dataset_info, mode, root)

    self._dataset = tf.data.TFRecordDataset(file_names,
                                            num_parallel_reads=num_threads)

    self._dataset = self._dataset.prefetch(buffer_size)
    self._dataset = self._dataset.batch(parse_batch_size)
    self._dataset = self._dataset.map(self._parse_record,
                                      num_parallel_calls=num_threads)
    self._dataset = self._dataset.apply(tf.contrib.data.unbatch())


  def _parse_record(self, raw_data):
    """Parses the data into tensors."""
    print('parse_record')
    example = tf.parse_example(raw_data, self._feature_map)
    # Get all indices
    indices = tf.range(0, self._dataset_info.sequence_size)
    frames = example['frames']
    cameras = example['cameras']
    return frames, cameras

  def _get_randomized_indices(self):
    """Generates randomized indices into a sequence of a specific length."""
    indices = tf.range(0, self._dataset_info.sequence_size)
    indices = tf.random_shuffle(indices)
    indices = tf.slice(indices, begin=[0], size=[self._example_size])
    return indices

  def _preprocess_frames(self, example, indices):
    """Preprocesses the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    frames = tf.gather(frames, indices, axis=1)
    frames = tf.map_fn(
      _convert_frame_data, tf.reshape(frames, [-1]),
      dtype=tf.float32, back_prop=False)
    dataset_image_dimensions = tuple(
      [self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    frames = tf.reshape(
      frames, (-1, self._example_size) + dataset_image_dimensions)
    if (self._custom_frame_size and
        self._custom_frame_size != self._dataset_info.frame_size):
      frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
      new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
      frames = tf.image.resize_bilinear(
        frames, new_frame_dimensions[:2], align_corners=True)
      frames = tf.reshape(
        frames, (-1, self._example_size) + new_frame_dimensions)
    return frames

  def _preprocess_cameras(self, example, indices):
    """Preprocesses the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
      raw_pose_params,
      [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    cameras = tf.concat(
      [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
    return cameras

  # The following four methods are needed to implement a tf.data.Dataset
  # Delegate them to the dataset we create internally
  def _as_variant_tensor(self):
    return self._dataset._as_variant_tensor()

  @property
  def output_classes(self):
    return self._dataset.output_classes

  @property
  def output_shapes(self):
    return self._dataset.output_shapes

  @property
  def output_types(self):
    return self._dataset.output_types


def gqn_input_fn(
    dataset,
    context_size,
    root,
    mode,
    batch_size=1,
    num_epochs=1,
    # Optionally reshape frames
    custom_frame_size=None,
    # Queue params
    num_threads=4,
    buffer_size=256,
    seed=None):
  """
  Creates a tf.data.Dataset based op that returns data.
    Args:
      dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
          'rooms_free_camera_no_object_rotations',
          'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
          'shepard_metzler_7_parts'].
      context_size: integer, number of views to be used to assemble the context.
      root: string, path to the root folder of the data.
      mode: one of tf.estimator.ModeKeys.
      batch_size: (optional) batch size, defaults to 1.
      num_epochs: (optional) number of times to go through the dataset,
          defaults to 1.
      custom_frame_size: (optional) integer, required size of the returned
          frames, defaults to None.
      num_threads: (optional) integer, number of threads used to read and parse
          the record files, defaults to 4.
      buffer_size: (optional) integer, capacity of the underlying prefetch or
          shuffle buffer, defaults to 256.
      seed: (optional) integer, seed for the random number generators used in
          the dataset.

    Raises:
      ValueError: if the required version does not exist; if the required mode
         is not supported; if the requested context_size is bigger than the
         maximum supported for the given dataset version.
  """

  if mode == tf.estimator.ModeKeys.TRAIN:
    str_mode = 'train'
  else:
    str_mode = 'test'

  dataset_path = dataset
  dataset = GQNTFRecordDataset(
      dataset, root, str_mode, custom_frame_size, num_threads,
      buffer_size)

  dataset = dataset.prefetch(buffer_size * batch_size)

  current_record = {
    'frames': [],
    'cameras': []
  }
  target_dir = '/n/scanner/datasets/gqn-dataset-pytorch'
  record_batch_size = 5000
  current_record_size = 0
  record_offset = 0
  os.makedirs(os.path.join(target_dir, dataset_path, str_mode), exist_ok=True)
  for i, (frames, cameras) in enumerate(dataset):
    if current_record_size == record_batch_size:
      record_path = os.path.join(
        target_dir, dataset_path, str_mode,
        'record-{:04d}.bin'.format(record_offset))
      with open(record_path, 'wb') as f:
        f.write(datasets.common.serialize_record(current_record))
      current_record['frames'].clear()
      current_record['cameras'].clear()
      current_record_size = 0
      record_offset += 1
    current_record['frames'].append(frames.numpy())
    current_record['cameras'].append(cameras.numpy())
    current_record_size += 1

  # Write last record
  if current_record_size > 0:
    record_path = os.path.join(
      target_dir, dataset_path, str_mode,
      'record-{:04d}.bin'.format(record_offset))
    with open(record_path, 'wb') as f:
      f.write(datasets.common.serialize_record(current_record))

  print('finished')


def main():
  data_dir = '/n/scanner/datasets/gqn-dataset'
  dataset = 'rooms_ring_camera'
  batch_size = 36
  num_threads = 4
  buffer_size = 4

  gqn_input_fn(
    dataset=dataset,
    context_size=-1,
    root=data_dir,
    mode=tf.estimator.ModeKeys.TRAIN,
    batch_size=batch_size,
    num_threads=num_threads,
    buffer_size=buffer_size)


if __name__ == '__main__':
  main()
