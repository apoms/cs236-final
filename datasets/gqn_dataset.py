from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import datasets.common
import os
import io
import torchvision.transforms

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


class GQNDataset(Dataset):
    def __init__(self, root_dir, dataset, mode='train'):
        # Read all files paths
        self.dataset_info = datasets.common._DATASETS[dataset]
        self.paths = get_pytorch_dataset_files(self.dataset_info, mode, root_dir)

        # Figure out how many examples in all records
        with open(self.paths[0], 'rb') as f:
            data = f.read()
            record = datasets.common.deserialize_record(data)
        self._examples_per_record = len(record['frames'])
        # Last record might contain less than usual
        with open(self.paths[-1], 'rb') as f:
            data = f.read()
            record = datasets.common.deserialize_record(data)
        self._examples_in_last_record = len(record['frames'])

        self._total_examples = (
            self._examples_per_record * (len(self.paths) - 1) +
            self._examples_in_last_record)
        print('Total examples: ', self._total_examples)

    def __len__(self):
        return self._total_examples

    def __getitem__(self, idx):
        record_idx = idx // self._examples_per_record
        record_local_idx = idx - self._examples_per_record * record_idx
        with open(self.paths[record_idx], 'rb') as f:
            record = datasets.common.read_record(f, [record_local_idx])

        frames, cameras = record['frames'][0], record['cameras'][0]
        new_frames = []
        for frame in frames:
            image = torchvision.transforms.ToTensor()(
                Image.open(io.BytesIO(frame)))
            new_frames.append(image)
        new_frames = np.stack(new_frames, axis=0)
        # Preprocess cameras
        new_cameras = cameras.reshape(
            (self.dataset_info.sequence_size,
             datasets.common._NUM_RAW_CAMERA_PARAMS))
        pos = new_cameras[:, 0:3]
        yaw = new_cameras[:, 3:4]
        pitch = new_cameras[:, 4:5]
        roll = np.zeros_like(pitch)
        new_cameras = np.concatenate(
            (pos, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)),
            axis=1)
        # new_cameras = np.concatenate(
        #     (pos, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch), np.sin(roll), np.cos(roll)),
        #     axis=1)
        new_cameras = new_cameras[:,:,np.newaxis,np.newaxis]
        return new_frames, new_cameras
