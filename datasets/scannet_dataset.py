from torch.utils.data import Dataset
from PIL import Image
from datasets.SensorData import SensorData, RGBDFrame

import numpy as np
import datasets.common
import os
import io
import torchvision.transforms
import sys
import struct
import time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_INFO = {
    'train_scans_path': os.path.join(SCRIPT_DIR, 'scannetv2_train.txt'),
    'val_scans_path': os.path.join(SCRIPT_DIR, 'scannetv2_val.txt'),
    'test_scans_path': os.path.join(SCRIPT_DIR, 'scannetv2_test.txt'),
    'num_scans': 1513
}


def read_scans_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def get_pytorch_dataset_files(dateset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath, mode)
  if mode == 'train':
    num_files = dateset_info.train_size
  else:
    num_files = dateset_info.test_size

  length = len(str(num_files))
  template = 'record-{:04d}.bin'
  # new tfrecord indexing runs from 1 to n
  return [os.path.join(base, template.format(i, num_files))
          for i in range(num_files)]


def build_scannet_index(root_dir, scan):
    index_path = os.path.join(root_dir, 'scans', scan,
                              scan + '_index.bin')
    if os.path.exists(index_path):
        return

    scan_path = os.path.join(root_dir, 'scans', scan, scan + '.sens')
    sensor_data = SensorData()
    frame_offsets = sensor_data.load_frame_offsets(scan_path)
    b = bytearray()
    b.extend(struct.pack('=Q', len(frame_offsets)))
    for fo in frame_offsets:
        b.extend(struct.pack('=Q', fo))
    with open(index_path, 'wb') as f:
        f.write(b)


def read_scannet_index(root_dir, scan):
    index_path = os.path.join(root_dir, 'scans', scan,
                              scan + '_index.bin')
    with open(index_path, 'rb') as f:
        data = f.read()
    num_frames, = struct.unpack('=Q', data[0:8])
    frame_offsets = []
    for i in range(num_frames):
        offset, = struct.unpack('=Q', data[8 + i * 8: 8 + (i + 1) * 8])
        frame_offsets.append(offset)
    return frame_offsets


class ScanNetDataset(Dataset):
    def __init__(self, root_dir, context_views=8, mode='train'):
        self.frame_interval = 15
        self.context_views = context_views
        # Read all scan paths
        self.root_dir = root_dir
        self.num_scans = DATASET_INFO['num_scans']
        if mode == 'train':
            scans_path = DATASET_INFO['train_scans_path']
        elif mode == 'val':
            scans_path = DATASET_INFO['val_scans_path']
        else:
            scans_path = DATASET_INFO['test_scans_path']

        self._scans_list = read_scans_list(scans_path)
        self._scans_list = ['scene{:04d}_00'.format(i) for i in range(420)]
        self._scan_indexes = {}
        self._scan_pairs = []
        for scan in self._scans_list:
            scan_path = os.path.join(self.root_dir, 'scans', scan,
                                     scan + '.sens')
            sensor_data = SensorData()
            sensor_data.load_num_frames(scan_path)
            num_frames = sensor_data.num_frames
            offset = 0
            while offset + self.frame_interval * context_views < num_frames:
                self._scan_pairs.append((scan, offset))
                offset += self.frame_interval * context_views
            build_scannet_index(self.root_dir, scan)
            self._scan_indexes[scan] = read_scannet_index(self.root_dir, scan)
        print('Total scenes: ', len(self._scans_list))
        print('Total examples: ', len(self._scan_pairs))

    def __len__(self):
        return len(self._scan_pairs)

    def __getitem__(self, idx):
        start_load = time.time()
        scan, frame_offset = self._scan_pairs[idx]
        scan_path = os.path.join(self.root_dir, 'scans', scan, scan + '.sens')
        sensor_data = SensorData()
        idxs = [frame_offset + i * self.frame_interval
                for i in range(self.context_views)]
        start_load_frames = time.time()
        rgbd_frames = sensor_data.load_frames(
            scan_path, self._scan_indexes[scan], idxs)
        end_load_frames = time.time()
        frames = []
        cameras = []
        decompress_frames_time = 0
        convert_frames_time = 0
        process_camera_time = 0
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
        ])
        for rgbd_frame in rgbd_frames:
            # Decompress frame
            decompress_frames_start = time.time()
            frame = Image.open(io.BytesIO(rgbd_frame.color_data))
            decompress_frames_time = time.time() - decompress_frames_start

            convert_frames_start = time.time()
            image = transforms(frame)
            convert_frames_time += time.time() - convert_frames_start
            # Process camera
            camera_time = time.time()
            R = np.linalg.inv(rgbd_frame.camera_to_world[0:3, 0:3])
            t = -rgbd_frame.camera_to_world[0:3, 2]
            # Convert rotation matrix to yaw, pitch, and roll
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
            roll = np.arctan2(R[2, 1], R[2, 2])
            if yaw != yaw:
                t = [0, 0, 0]
                yaw = 0
                pitch = 0
                roll = 0
            cameras.append(np.array(
                [t[0], t[1], t[2], np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch),
                 np.sin(roll), np.cos(roll)], dtype=np.float32))
            process_camera_time += time.time() - camera_time
            frames.append(image)

        new_frames = frames
        new_frames = np.stack(new_frames, axis=0)
        # Preprocess cameras
        new_cameras = np.stack(cameras, axis=0)
        new_cameras = new_cameras[:,:,np.newaxis,np.newaxis]

        end_load = time.time()

        if False:
            print('Time to load idx {:d}: {:f}'.format(idx, end_load - start_load))
            print('Time to read frames: {:f}'.format(end_load_frames - start_load_frames))
            print('Time to decompress frames: {:f}'.format(decompress_frames_time))
            print('Time to convert frames: {:f}'.format(convert_frames_time))
            print('Time to process camera: {:f}'.format(process_camera_time))

        return new_frames, new_cameras
