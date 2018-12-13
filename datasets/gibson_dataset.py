from torch.utils.data import Dataset
from PIL import Image
from datasets.common import Context, Query, TaskData

import numpy as np
import datasets.common
import os
import io
import torchvision.transforms
import collections
import json
import imageio
import PIL
import csv

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['split_path', 'frame_size']
)

_DATASETS = dict(
  tiny=DatasetInfo(
    split_path='splits/train_val_test_tiny.csv',
    frame_size=512
  ),
  full=DatasetInfo(
    split_path='splits/train_val_test_full.csv',
    frame_size=512
  )
)


def parse_split_file(split_path):
    train_models = []
    val_models = []
    test_models = []
    with open(split_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if header is None:
                header = row
                continue
            name = row[0]
            if row[1] == '1':
                train_models.append(name)
            elif row[2] == '1':
                val_models.append(name)
            elif row[3] == '1':
                test_models.append(name)
    return train_models, val_models, test_models


def get_dataset_data(dateset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  base = root
  split_path = dateset_info.split_path
  full_split_path = os.path.join(root, split_path)
  train_models, val_models, test_models = parse_split_file(full_split_path)
  if mode == 'train':
    names = train_models
  elif mode == 'val':
    names = val_models
  else:
    names = test_models

  cam_id_template = '{:05d}'
  data_tuples = []
  for name in names:
    dir_path = os.path.join(base, name)
    metadata_path = os.path.join(dir_path, 'views', 'metadata.json')
    if not os.path.exists(metadata_path):
      continue
    with open(metadata_path, 'r') as f:
      metadata = json.load(f)
    for cam_id, data in metadata.items():
      num_images = data['num_images']
      if num_images < 20:
        continue
      cam_id_prefix = cam_id_template.format(int(cam_id))

      pose_path = os.path.join(dir_path, 'views', cam_id_prefix + '_poses.txt')
      with open(pose_path, 'r') as f:
          x = len(f.readlines())
      if x < 21:
          continue
      view_template = os.path.join(dir_path, 'views',
                                   cam_id_prefix + '_{:06d}.png')
      data_tuple = (pose_path, view_template, num_images)
      data_tuples.append(data_tuple)
  return data_tuples


class GibsonDataset(Dataset):
    def __init__(self, root_dir, dataset='tiny', mode='train',
                 resize_shape=(256, 256)):
        # Read all files paths
        self.dataset_info = _DATASETS[dataset]
        self.data_tuples = get_dataset_data(self.dataset_info, mode, root_dir)
        self._resize_shape = resize_shape

        # Figure out how many examples in all records
        self._total_examples = len(self.data_tuples)
        print('Total examples: ', self._total_examples)

    def __len__(self):
        return self._total_examples

    def __getitem__(self, idx):
        pose_path, view_template, num_images = self.data_tuples[idx]
        # Read all frames and poses
        frames = []
        for i in range(num_images):
          frames.append(imageio.imread(view_template.format(i)))
        cameras = []
        with open(pose_path, 'r') as f:
          for line in f.readlines():
            view = np.zeros((1,7))
            values = [float(v) for v in line.split(' ')]
            for i, v in enumerate(values):
              view[0, i] = v
            cameras.append(view)

        new_frames = []
        for frame in frames:
            image = torchvision.transforms.ToTensor()(
              torchvision.transforms.Resize(self._resize_shape)(
                PIL.Image.fromarray(frame)))
            new_frames.append(image)
        new_frames = np.stack(new_frames, axis=0).astype(np.float32)
        # Preprocess cameras
        new_cameras = np.vstack(cameras[1:]).astype(np.float32)
        # Subtract out the reference position
        new_cameras[:, 0:3] -= cameras[0][:, 0:3]
        new_cameras = new_cameras[:,:,np.newaxis,np.newaxis]
        return new_frames, new_cameras
