from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import struct
import numpy as np

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    # super-small subset of rooms_ring for debugging purposes
    # TODO(ogroth): provide dataset
    rooms_ring_camera_debug=DatasetInfo(
        basepath='rooms_ring_camera_debug',
        train_size=1,  # 18
        test_size=1,  # 2
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _get_dataset_files(dateset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath, mode)
  if mode == 'train':
    num_files = dateset_info.train_size
  else:
    num_files = dateset_info.test_size

  length = len(str(num_files))
  template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
  # new tfrecord indexing runs from 1 to n
  return [os.path.join(base, template.format(i, num_files))
          for i in range(1, num_files + 1)]


def serialize_record(record):
  num_items = len(record['frames'])
  b = bytearray()
  # Write data header
  b.extend(struct.pack('=Q', num_items))
  num_frames = record['frames'][0].shape[0]
  b.extend(struct.pack('=Q', num_frames))
  # Write offset of each item
  offset = 0
  for i in range(num_items):
    b.extend(struct.pack('=Q', offset))
    total_size = 0
    frames = record['frames'][i]
    cameras = record['cameras'][i]
    for j in range(num_frames):
      total_size += 8
      total_size += len(frames[j])
    total_size += 8
    total_size += len(cameras.tobytes())
    offset += total_size
  # Write data
  for i in range(num_items):
    frames = record['frames'][i]
    cameras = record['cameras'][i]
    for j in range(num_frames):
      b.extend(struct.pack('=Q', len(frames[j])))
      b.extend(frames[j])
    b.extend(struct.pack('=Q', len(cameras.tobytes())))
    b.extend(cameras.tobytes())
  return b


def deserialize_record(data):
  record = {'frames': [], 'cameras': []}
  offset = 0
  def extract(size):
    nonlocal offset
    d = data[offset:offset+size]
    offset += size
    return d
  num_items, = struct.unpack('=Q', extract(8))
  num_frames, = struct.unpack('=Q', extract(8))
  # Ignore header
  offset += num_items * 8
  # Read data
  for i in range(num_items):
    example_frames = []
    for j in range(num_frames):
      frame_size, = struct.unpack('=Q', extract(8))
      example_frames.append(extract(frame_size))
    camera_size, = struct.unpack('=Q', extract(8))
    example_cameras = np.frombuffer(extract(camera_size), dtype=np.float32)
    record['frames'].append(example_frames)
    record['cameras'].append(example_cameras)
  return record


def read_record(f, idxs):
  record = {'frames': [], 'cameras': []}
  offset = 0

  def read(of, size):
    f.seek(of)
    return f.read(size)

  def extract(size):
    nonlocal offset
    d = read(offset, size)
    offset += size
    return d

  num_items, = struct.unpack('=Q', extract(8))
  num_frames, = struct.unpack('=Q', extract(8))
  # Find offsets for requested idxs
  idxs_offsets = []
  for idx in idxs:
    o = offset + idx * 8
    off, = struct.unpack('=Q', read(o, 8))
    idxs_offsets.append(off)
  offset += num_items * 8
  base_offset = int(offset)
  # Read data
  for off in idxs_offsets:
    offset = base_offset + off
    example_frames = []
    for j in range(num_frames):
      frame_size, = struct.unpack('=Q', extract(8))
      example_frames.append(extract(frame_size))
    camera_size, = struct.unpack('=Q', extract(8))
    example_cameras = np.frombuffer(extract(camera_size), dtype=np.float32)
    record['frames'].append(example_frames)
    record['cameras'].append(example_cameras)
  return record
