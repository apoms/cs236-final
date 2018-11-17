# Taken from: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# Modified by apoms

import os, struct
import numpy as np
import zlib
import imageio
import cv2

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  def load(self, file_handle, read_depth=False):
    data = file_handle.read(16*4 + 8 * 4)
    unpacked = struct.unpack('f'*16, data[0:16*4])
    self.camera_to_world = np.asarray(unpacked, dtype=np.float32).reshape(4, 4)
    offset = 16*4
    self.timestamp_color = struct.unpack('Q', data[offset:offset+8])[0]
    offset += 8
    self.timestamp_depth = struct.unpack('Q', data[offset:offset+8])[0]
    offset += 8
    self.color_size_bytes = struct.unpack('Q', data[offset:offset+8])[0]
    offset += 8
    self.depth_size_bytes = struct.unpack('Q', data[offset:offset+8])[0]
    offset += 8
    self.color_data = file_handle.read(self.color_size_bytes)
    if read_depth:
        self.depth_data = file_handle.read(self.depth_size_bytes)


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename=None):
    self.version = 4
    if filename:
        self.load(filename)

  def _read_header(self, f):
    version = struct.unpack('I', f.read(4))[0]
    assert self.version == version
    strlen = struct.unpack('Q', f.read(8))[0]
    self.sensor_name = str(struct.unpack('c'*strlen, f.read(strlen)))
    self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
    self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
    self.color_width = struct.unpack('I', f.read(4))[0]
    self.color_height =  struct.unpack('I', f.read(4))[0]
    self.depth_width = struct.unpack('I', f.read(4))[0]
    self.depth_height =  struct.unpack('I', f.read(4))[0]
    self.depth_shift =  struct.unpack('f', f.read(4))[0]
    self.num_frames =  struct.unpack('Q', f.read(8))[0]

  def load_header(self, filename):
    with open(filename, 'rb') as f:
      self._read_header(f)

  def load(self, filename):
    with open(filename, 'rb') as f:
      self._read_header(f)
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)

  def load_frames(self, filename, index, idxs):
    frames = []
    with open(filename, 'rb') as f:
      self._read_header(f)
      self.frames = []
      for i in idxs:
          offset = index[i]
          f.seek(offset)
          frame = RGBDFrame()
          frame.load(f)
          frames.append(frame)
    return frames

  def load_num_frames(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      offset = 4 + 8 + strlen + (16 * 4) * 4 + 4 * 7
      f.seek(offset)
      self.num_frames =  struct.unpack('Q', f.read(8))[0]

  def load_frame_offsets(self, filename):
    offsets = []
    with open(filename, 'rb') as f:
      self._read_header(f)
      self.frames = []
      offset = f.tell()
      for i in range(self.num_frames):
          f.seek(16 * 4 + 8 * 2, os.SEEK_CUR)
          color_size_bytes = struct.unpack('Q', f.read(8))[0]
          depth_size_bytes = struct.unpack('Q', f.read(8))[0]
          f.seek(color_size_bytes + depth_size_bytes, os.SEEK_CUR)
          offsets.append(offset)
          offset = f.tell()
    return offsets

  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)


  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting camera intrinsics to', output_path)
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))
