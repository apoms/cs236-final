from datasets.gqn_dataset import GQNDataset
from datasets.scannet_dataset import ScanNetDataset
from torch.utils.data import DataLoader
from gqn_deterministic import Renderer
from gqn_encoder import Tower
from collections import defaultdict

import numpy as np
import time
import torch
import gqn_train
import argparse
import random
import sys
import scipy.misc


def main(args):
    # Load the model and a random scene
    print('Loading model and dataset...')
    dataset = 'rooms_ring_camera'
    train_dataset = GQNDataset(
        root_dir=args.data_dir,
        dataset=dataset,
        mode='train')
    test_dataset = GQNDataset(
        root_dir=args.data_dir,
        dataset=dataset,
        mode='test')
    pose_channels = 7

    rep_net = Tower(pose_channels)

    # Pass dummy data through the network to get the representation
    # shape for the Generator and Inference networks
    dummy_images = torch.rand(1, 3, 64, 64)
    dummy_poses = torch.rand(1, pose_channels, 1, 1)
    dummy_rep = rep_net(dummy_images, dummy_poses)

    model_settings = {
        'fn_generator_type': 'DRAW',
    }
    renderer = Renderer(dummy_rep.shape, pose_channels, model_settings,
                        train=False)

    if args.cuda:
        rep_net = rep_net.cuda()
        renderer = renderer.cuda()

    seed = random.randrange(sys.maxsize)
    step = 0

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print('Resuming from {:s}...'.format(checkpoint_path))
        checkpoint = gqn_train.read_checkpoint(checkpoint_path)
        #seed = checkpoint['seed']
        rep_net.load_state_dict(checkpoint['representation_net'])
        renderer.load_state_dict(checkpoint['renderer_net'])
        resume_step = checkpoint['step']

    torch.manual_seed(seed)
    random.seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=True, num_workers=4)

    num_context_views = 3
    num_query_views = 10 - num_context_views
    num_examples = 5
    display_dim = (num_examples * 2, num_context_views + 1 + num_query_views)
    num_frames = 480
    saved_frames = defaultdict(dict)
    context_frames = {}
    query_frames = {}
    rendered_context_frames = defaultdict(dict)
    rendered_query_frames = defaultdict(dict)
    for i in range(num_examples):
        frames, cameras = next(iter(train_dataloader))

        print('Generating representation...')
        batch_size = frames.shape[0]

        b = np.random.random(frames.shape[0:2])
        idxs = np.argsort(b, axis=-1)
        input_idx = idxs[:, :num_context_views]
        query_idx = idxs[:, num_context_views:]
        t = np.arange(frames.shape[0])[:,None]

        input_images = frames[t, input_idx, ...]
        input_poses = cameras[t, input_idx, ...]

        query_images = frames[t, query_idx, ...]
        query_poses = cameras[t, query_idx, ...]

        query_pose = cameras[t, 0, ...]
        query_pose = torch.squeeze(query_pose, dim=1)

        context = {}
        for j in range(num_context_views):
            obs = input_images[0,j,:,:,:].detach().cpu().numpy()
            obs = 255 * obs / obs.max()
            obs = obs.astype(int)
            obs = np.swapaxes(obs, 0, 2)
            obs = np.swapaxes(obs, 0, 1)
            context[j] = obs
        context_frames[i] = context

        query = {}
        for j in range(num_query_views):
            obs = query_images[0,j,:,:,:].detach().cpu().numpy()
            obs = 255 * obs / obs.max()
            obs = obs.astype(int)
            obs = np.swapaxes(obs, 0, 2)
            obs = np.swapaxes(obs, 0, 1)
            query[j] = obs
        query_frames[i] = query

        if args.cuda:
            input_images = input_images.cuda()
            input_poses = input_poses.cuda()
            query_pose = query_pose.cuda()

        encode_start = time.time()
        rep = gqn_train.encode_representation(rep_net, input_images, input_poses)
        encode_end = time.time()

        posterior_global_z_dist = renderer.z_distribution(rep)
        global_z = posterior_global_z_dist.rsample()


        pos = 0
        pitch = 0
        yaw = 0
        r = 0
        for fi in range(num_frames):
            with torch.no_grad():
                r += (2 * np.pi) / num_frames
                pos = r
                yaw = -r

                pose_cpu = query_pose.cpu().numpy()
                pose_cpu[0, 0:2, 0, 0] = np.array(
                    [-np.cos(pos), np.sin(pos)],
                    dtype=np.float64)
                pose_cpu[0, 3:7, 0, 0] = np.array(
                    [np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)],
                    dtype=np.float64)
                query_pose = torch.from_numpy(pose_cpu)
                if args.cuda:
                    query_pose = query_pose.cuda()

                u_g = renderer.forward(global_z, query_pose, None, args.cuda, False)
                d_obs = renderer.observation_distribution(u_g, 0)
                obs = d_obs.mean[0,:,:,:].detach().cpu().numpy()
                obs = 255 * obs / obs.max()
                obs = obs.astype(int)
                obs = np.swapaxes(obs, 0, 2)
                obs = np.swapaxes(obs, 0, 1)
                saved_frames[i][fi] = obs

        def render_image(query_pose):
            with torch.no_grad():
                if args.cuda:
                    query_pose = query_pose.cuda()
                u_g = renderer.forward(global_z, query_pose, None, args.cuda, False)
                d_obs = renderer.observation_distribution(u_g, 0)
                obs = d_obs.mean[0,:,:,:].detach().cpu().numpy()
                obs = 255 * obs / obs.max()
                obs = obs.astype(int)
                obs = np.swapaxes(obs, 0, 2)
                obs = np.swapaxes(obs, 0, 1)
                return obs

        context = {}
        for j in range(input_poses.shape[1]):
            context[j] = render_image(input_poses[:, j, ...])
        rendered_context_frames[i] = context

        query = {}
        for j in range(query_poses.shape[1]):
            query[j] = render_image(query_poses[:, j, ...])
        rendered_query_frames[i] = query

    if True:
        # Draw context images side by side with rendered view
        shape = (display_dim[0] * 64, display_dim[1] * 64, 3)
        mega_frame = np.zeros(shape, dtype=np.int)
        for i in range(num_examples):
            for v in range(num_context_views):
                x = v
                y = i
                xoff = x * 64
                yoff = (y * 64) * 2
                mega_frame[yoff:yoff+64, xoff:xoff+64, :] = context_frames[i][v]
                mega_frame[yoff+64:yoff+128, xoff:xoff+64, :] = rendered_context_frames[i][v]
        # Draw query images
        for i in range(num_examples):
            for v in range(num_query_views):
                x = v + num_context_views + 1
                y = i
                xoff = x * 64
                yoff = (y * 64) * 2
                mega_frame[yoff:yoff+64, xoff:xoff+64, :] = query_frames[i][v]
                mega_frame[yoff+64:yoff+128, xoff:xoff+64, :] = rendered_query_frames[i][v]
        # Draw rendered view
        for fi in range(num_frames):
            for i in range(num_examples):
                x = num_context_views
                y = i
                xoff = x * 64
                yoff = (y * 64) * 2
                mega_frame[yoff:yoff+64, xoff:xoff+64, :] = saved_frames[i][fi]

            scipy.misc.imsave('mega{:d}.png'.format(fi), mega_frame)
    elif False:
        # Draw rectangular grid 
        shape = (display_dim[0] * 64, display_dim[1] * 64, 3)
        mega_frame = np.zeros(shape, dtype=np.int)
        for fi in range(num_frames):
            for i in range(num_examples):
                x = i % display_dim[0]
                y = i // display_dim[0]
                xoff = x * 64
                yoff = y * 64
                mega_frame[yoff:yoff+64, xoff:xoff+64, :] = saved_frames[i][fi]

            scipy.misc.imsave('mega{:d}.png'.format(fi), mega_frame)


if __name__ == '__main__':
    # command line argument parser
    parser = argparse.ArgumentParser(
        description='Play with a GQN using Pytorch.')
    # directory parameters
    parser.add_argument(
        '--data-dir', type=str, default='/tmp/data/gqn-dataset',
        help='The path to the gqn-dataset-pytorch directory.')
    parser.add_argument(
        '--checkpoint', type=str,
        help='The checkpoint file to resume from.')
    # training parameters
    parser.add_argument(
        '--cuda', default=False, action='store_true',
        help="Run the model using CUDA.")

    args = parser.parse_args()

    main(args)
