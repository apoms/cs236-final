from __future__ import print_function

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from gqn_generator import FunctionGenerator, Generator, Inference
import gqn_deterministic 
import proj_net 
from gqn_encoder import Tower
from datasets.gqn_dataset import GQNDataset
from datasets.scannet_dataset import ScanNetDataset
from datasets.gibson_dataset import GibsonDataset
from torch.utils.data import DataLoader

import torch
import torch.utils.data
import numpy as np
import scipy.misc
import argparse
import random
import sys
import os
import time
import datetime
import tensorboardX
import torchvision.utils

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D

LSTM_LAYERS = 8

PIXEL_ANNEALING_ITERS = 2e5
PIXEL_SIGMA_INITIAL = 2.0
PIXEL_SIGMA_FINAL = 0.7

LR_ANNEALING_ITERS = 1.6e6
LR_INITIAL = 2e-4
LR_FINAL = 2e-5


def calculate_pixel_sigma(step):
    adaptive_sigma = ((PIXEL_SIGMA_INITIAL - PIXEL_SIGMA_FINAL) *
                      (1 - step / PIXEL_ANNEALING_ITERS))
    return max(PIXEL_SIGMA_FINAL + adaptive_sigma, PIXEL_SIGMA_FINAL)


def calculate_lr(step):
    adaptive_lr = ((LR_INITIAL - LR_FINAL) *
                   (1 - step / LR_ANNEALING_ITERS))
    return max(LR_FINAL + adaptive_lr, LR_FINAL)


def get_output_dir(run_dir):
    return os.path.join(run_dir, 'outputs')


def get_checkpoint_dir(run_dir):
    return os.path.join(run_dir, 'checkpoints')


def get_checkpoint_path(run_dir, step):
    return os.path.join(get_checkpoint_dir(run_dir),
                        'model-{:010d}.pth'.format(step))


def write_checkpoint(path, data):
    torch.save(data, path)


def read_checkpoint(path):
    return torch.load(path)


def encode_representation(net, images, poses):
    reps = []
    for i in range(images.shape[1]):
        rep = net(images[:, i, ...], poses[:, i, ...])
        reps.append(rep[:,None,...])
    rep = torch.sum(torch.cat(reps, dim=1), dim=1)
    return rep


def gqn_train_iter(args):
    step = args['step']
    dp_rep_net = args['dp_rep_net']
    rep_net = args['rep_net']
    renderer = args['renderer']
    optimizer = args['optimizer']
    input_images = args['input_images']
    input_poses = args['input_poses']
    all_images = args['all_images']
    all_poses = args['all_poses']
    query_images = args['query_images']
    query_poses = args['query_poses']
    writer = args['writer']
    log_steps = args['log_steps']
    checkpoint_steps = args['checkpoint_steps']
    run_path = args['run_path']

    # Test time representation
    encode_start = time.time()
    input_rep = encode_representation(dp_rep_net, input_images, input_poses)
    encode_end = time.time()

    # Full representation
    encode_query_start = time.time()
    all_rep = encode_representation(dp_rep_net, all_images, all_poses)
    encode_query_end = time.time()

    # Generator distribution over functions
    fn_gen_start = time.time()
    all_global_z_dist, kls = renderer.z_distribution_with_loss(input_rep, all_rep)
    fn_gen_end = time.time()

    # Sample z vector representing the global variations in the data
    global_z = all_global_z_dist.rsample()


    kl = 0
    log_likelihood = 1
    for qi in range(num_query_views):
        query_image = query_images[:, qi, ...]
        query_pose = query_poses[:, qi, ...]

        decode_start = time.time()
        u_g = renderer.forward(global_z, query_pose, query_image, cuda,
                               args.debug)
        decode_end = time.time()

        # Sum up all the kl divergences for each rendering step
        #for i in range(1, len(kl_divs)):
        #    kl += kl_divs[i]

        # Compute generator observation distribution and calculate
        # probablity of actual observation under that distribution
        observation_sigma = calculate_pixel_sigma(step)
        d_obs = renderer.observation_distribution(u_g, observation_sigma)
        # ELBO reconstruction loss
        log_likelihood += torch.sum(
            torch.mean(d_obs.log_prob(query_image), dim=0))

    #KL_loss = torch.sum(torch.mean(kl, dim=0)) / num_query_views
    log_likelihood /= num_query_views

    kl = kls[0]
    for i in range(1, len(kls)):
        kl += kls[i]
    kl_z = torch.sum(torch.mean(kl, dim=0))

    #ELBO = torch.sum(torch.mean(ll_input_z - ll_all_z, dim=0))
    ELBO = 0
    ELBO += kl_z
    #ELBO += KL_loss
    ELBO -= log_likelihood

    backward_start = time.time()

    loss = ELBO.item()
    ELBO.backward()
    optimizer.step()

    backward_end = time.time()

    # Update optimizer learning rate
    new_lr = calculate_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    if args.debug:
        print('Num context views: {:d}'.format(num_context_views))
        print('Encode time: {:.2f}'.format(encode_end - encode_start))
        print('Fn Gen time: {:.2f}'.format(fn_gen_end - fn_gen_start))
        print('Decode time: {:.2f}'.format(decode_end - decode_start))
        print('Backward time: {:.2f}'.format(
            backward_end - backward_start))
        print()

    elapsed_time = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=elapsed_time))
    if step % log_steps == 0:
        print('Logging step {:d} (epoch {:d}, {:d}/{:d})...'.format(
            step, e, i_batch, len(train_dataloader)))
        print('Elapsed time: {:s}'.format(formatted_time))
        print('Context views: ', num_context_views)
        print('Query views: ', num_query_views)
        #print('KL Loss: ', KL_loss)
        print('LL Loss: ', log_likelihood.cpu().data)
        print('KL z Loss: ', kl_z.cpu().data)
        print('loss: ', loss)
        print('sigma: ', observation_sigma)
        print('lr: ', new_lr)
        print('Writing image...')
        obs = d_obs.mean[0,:,:,:].detach().cpu().numpy()
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 0, 1)
        target = query_image[0,:,:,:].detach().cpu().numpy()
        target = np.swapaxes(target, 0, 2)
        target = np.swapaxes(target, 0, 1)
        img = np.concatenate((target, obs), axis=1)
        path = get_output_dir(args.run_dir)
        scipy.misc.imsave(
            os.path.join(path, 'sample{:06d}.png'.format(step)), img)
        writer.add_image(
            'data/obs_image',
            torchvision.utils.make_grid(
                d_obs.mean, normalize=True, scale_each=True),
            step)
        writer.add_image(
            'data/target_image', torchvision.utils.make_grid(query_image), step)
        #writer.add_scalar('data/kl_loss', KL_loss, step)
        writer.add_scalar('data/nll_loss', -log_likelihood.cpu().data, step)
        writer.add_scalar('data/kl_z_loss', kl_z.cpu().data, step)
        writer.add_scalar('data/loss', loss, step)
        writer.add_scalar('param/sigma', observation_sigma, step)
        writer.add_scalar('param/lr', new_lr, step)
        print('Done logging.')
        print()

    if step % args.checkpoint_steps == 0:
        print('Checkpointing step {:d}...'.format(step))
        print('Elapsed time: {:s}'.format(formatted_time))
        path = get_checkpoint_path(args.run_dir, step)
        data = {
            'optimizer': optimizer.state_dict(),
            'step': step,
            'representation_net': rep_net.state_dict(),
            'renderer_net': renderer.state_dict(),
            'dataset': args.dataset,
            'seed': seed,
            'total_time': elapsed_time,
        }
        write_checkpoint(path, data)
        print('Done checkpointing.')
        print()

    step += 1

    
def proj_train_iter(args):
    step = args['step']
    dp_rep_net = args['dp_rep_net']
    rep_net = args['rep_net']
    renderer = args['renderer']
    optimizer = args['optimizer']
    input_images = args['input_images']
    input_poses = args['input_poses']
    all_images = args['all_images']
    all_poses = args['all_poses']
    query_images = args['query_images']
    query_poses = args['query_poses']
    writer = args['writer']
    log_steps = args['log_steps']
    checkpoint_steps = args['checkpoint_steps']
    run_path = args['run_path']


    def encode_representations(net, images, poses):
        reps = []
        for i in range(images.shape[1]):
            rep = net(images[:, i, ...], poses[:, i, ...])
            reps.append(rep[:, None,...])
        return reps

    # Test time representation
    encode_start = time.time()
    input_reps = encode_representations(dp_rep_net, input_images, input_poses)
    encode_end = time.time()

    # Full representation
    encode_query_start = time.time()
    all_reps = encode_representations(dp_rep_net, all_images, all_poses)
    encode_query_end = time.time()

    batch_size = input_images.shape[0]
    grids = []
    for b in range(batch_size):
        shape = [32, 32, 32]
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
        grids.append(renderer.init_grid(shape, bounds))
    # Generator distribution over functions
    fn_gen_start = time.time()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    start = time.time()
    for i in range(len(all_reps)):
        renderer.trace(grids, all_reps[i], None, all_poses[:, i, ...], ax)
    print('time to trace all reps {:f}'.format(time.time() - start))
    plt.show()
    plt.savefig('3dfig.png')
    fn_gen_end = time.time()
    exit(0)

    # Sample z vector representing the global variations in the data
    global_z = all_global_z_dist.rsample()


    kl = 0
    log_likelihood = 1
    for qi in range(num_query_views):
        query_image = query_images[:, qi, ...]
        query_pose = query_poses[:, qi, ...]

        decode_start = time.time()
        u_g = renderer.forward(global_z, query_pose, query_image, cuda,
                               args.debug)
        decode_end = time.time()

        # Sum up all the kl divergences for each rendering step
        #for i in range(1, len(kl_divs)):
        #    kl += kl_divs[i]

        # Compute generator observation distribution and calculate
        # probablity of actual observation under that distribution
        observation_sigma = calculate_pixel_sigma(step)
        d_obs = renderer.observation_distribution(u_g, observation_sigma)
        # ELBO reconstruction loss
        log_likelihood += torch.sum(
            torch.mean(d_obs.log_prob(query_image), dim=0))

    #KL_loss = torch.sum(torch.mean(kl, dim=0)) / num_query_views
    log_likelihood /= num_query_views

    kl = kls[0]
    for i in range(1, len(kls)):
        kl += kls[i]
    kl_z = torch.sum(torch.mean(kl, dim=0))

    #ELBO = torch.sum(torch.mean(ll_input_z - ll_all_z, dim=0))
    ELBO = 0
    ELBO += kl_z
    #ELBO += KL_loss
    ELBO -= log_likelihood

    backward_start = time.time()

    loss = ELBO.item()
    ELBO.backward()
    optimizer.step()

    backward_end = time.time()

    # Update optimizer learning rate
    new_lr = calculate_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    if args.debug:
        print('Num context views: {:d}'.format(num_context_views))
        print('Encode time: {:.2f}'.format(encode_end - encode_start))
        print('Fn Gen time: {:.2f}'.format(fn_gen_end - fn_gen_start))
        print('Decode time: {:.2f}'.format(decode_end - decode_start))
        print('Backward time: {:.2f}'.format(
            backward_end - backward_start))
        print()

    elapsed_time = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=elapsed_time))
    if step % log_steps == 0:
        print('Logging step {:d} (epoch {:d}, {:d}/{:d})...'.format(
            step, e, i_batch, len(train_dataloader)))
        print('Elapsed time: {:s}'.format(formatted_time))
        print('Context views: ', num_context_views)
        print('Query views: ', num_query_views)
        #print('KL Loss: ', KL_loss)
        print('LL Loss: ', log_likelihood.cpu().data)
        print('KL z Loss: ', kl_z.cpu().data)
        print('loss: ', loss)
        print('sigma: ', observation_sigma)
        print('lr: ', new_lr)
        print('Writing image...')
        obs = d_obs.mean[0,:,:,:].detach().cpu().numpy()
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 0, 1)
        target = query_image[0,:,:,:].detach().cpu().numpy()
        target = np.swapaxes(target, 0, 2)
        target = np.swapaxes(target, 0, 1)
        img = np.concatenate((target, obs), axis=1)
        path = get_output_dir(args.run_dir)
        scipy.misc.imsave(
            os.path.join(path, 'sample{:06d}.png'.format(step)), img)
        writer.add_image(
            'data/obs_image',
            torchvision.utils.make_grid(
                d_obs.mean, normalize=True, scale_each=True),
            step)
        writer.add_image(
            'data/target_image', torchvision.utils.make_grid(query_image), step)
        #writer.add_scalar('data/kl_loss', KL_loss, step)
        writer.add_scalar('data/nll_loss', -log_likelihood.cpu().data, step)
        writer.add_scalar('data/kl_z_loss', kl_z.cpu().data, step)
        writer.add_scalar('data/loss', loss, step)
        writer.add_scalar('param/sigma', observation_sigma, step)
        writer.add_scalar('param/lr', new_lr, step)
        print('Done logging.')
        print()

    if step % args.checkpoint_steps == 0:
        print('Checkpointing step {:d}...'.format(step))
        print('Elapsed time: {:s}'.format(formatted_time))
        path = get_checkpoint_path(args.run_dir, step)
        data = {
            'optimizer': optimizer.state_dict(),
            'step': step,
            'representation_net': rep_net.state_dict(),
            'renderer_net': renderer.state_dict(),
            'dataset': args.dataset,
            'seed': seed,
            'total_time': elapsed_time,
        }
        write_checkpoint(path, data)
        print('Done checkpointing.')
        print()

    step += 1


def train(args):
    cuda = args.cuda
    BATCH_SIZE = args.batch_size

    writer = tensorboardX.SummaryWriter(
        log_dir=os.path.join(args.run_dir, 'tensorboard'))

    if args.dataset == 'gibson':
        train_dataset = GibsonDataset(
            root_dir=args.data_dir,
            dataset='tiny',
            mode='train')
        test_dataset = GibsonDataset(
            root_dir=args.data_dir,
            dataset='tiny',
            mode='test')
        pose_channels = 7
        image_shape = (256, 256)
    if args.dataset == 'gibson-full':
        image_shape = (128, 128)
        train_dataset = GibsonDataset(
            root_dir=args.data_dir,
            dataset='full',
            mode='train',
            resize_shape=image_shape)
        test_dataset = GibsonDataset(
            root_dir=args.data_dir,
            dataset='full',
            mode='test',
            resize_shape=image_shape)
        pose_channels = 7
    elif args.dataset == 'scannet':
        train_dataset = ScanNetDataset(
            root_dir=args.data_dir,
            mode='train')
        test_dataset = ScanNetDataset(
            root_dir=args.data_dir,
            mode='test')
        pose_channels = 9
    else:
        train_dataset = GQNDataset(
            root_dir=args.data_dir,
            dataset=args.dataset,
            mode='train')
        test_dataset = GQNDataset(
            root_dir=args.data_dir,
            dataset=args.dataset,
            mode='test')
        pose_channels = 7
        image_shape = (64, 64)

    if args.model == 'jump-vae' or args.model == 'jump-draw': 
        rep_net = Tower(pose_channels)

        # Pass dummy data through the network to get the representation
        # shape for the Generator and Inference networks
        dummy_images = torch.rand(1, 3, *image_shape)
        dummy_poses = torch.rand(1, pose_channels, 1, 1)
        dummy_rep = rep_net(dummy_images, dummy_poses)

        model_settings = {
            'fn_generator_type': 'DRAW' if args.model == 'jump-draw' else 'VAE',
        }
        renderer = gqn_deterministic.Renderer(dummy_rep.shape,
                                              pose_channels, model_settings)
        train_fn = gqn_train_iter
    elif args.model == 'proj': 
        rep_net = Tower(pose_channels)

        # Pass dummy data through the network to get the representation
        # shape for the Generator and Inference networks
        dummy_images = torch.rand(1, 3, *image_shape)
        dummy_poses = torch.rand(1, pose_channels, 1, 1)
        dummy_rep = rep_net(dummy_images, dummy_poses)

        renderer = proj_net.Renderer(image_shape, dummy_rep.shape,
                                     pose_channels, {})
        train_fn = proj_train_iter
    else:
        print('No model of type "{:s}" available.'.format(args.model))

    if args.data_parallel:
        dp_rep_net = nn.DataParallel(rep_net)
        dp_renderer = nn.DataParallel(renderer)
    else:
        dp_rep_net = rep_net
        dp_renderer = renderer

    if cuda:
        dp_rep_net = dp_rep_net.cuda()
        dp_renderer = dp_renderer.cuda()

    model_params = (
        list(dp_rep_net.parameters()) +
        list(dp_renderer.parameters()))
    optimizer = optim.Adam(model_params, lr=LR_INITIAL)

    seed = random.randrange(sys.maxsize)
    step = 0

    if args.resume_from:
        checkpoint_path = args.resume_from
        print('Resuming from {:s}...'.format(checkpoint_path))
        checkpoint = read_checkpoint(checkpoint_path)
        seed = checkpoint['seed']
        optimizer.load_state_dict(checkpoint['optimizer'])
        rep_net.load_state_dict(checkpoint['representation_net'])
        renderer.load_state_dict(checkpoint['renderer_net'])
        resume_step = checkpoint['step']

    torch.manual_seed(seed)
    random.seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

    start_epoch = 0
    if args.resume_from:
        if False:
            print('Advancing to step {:d}...'.format(resume_step))
            while True:
                for _, _ in train_dataloader:
                    step += 1
                    if step % args.checkpoint_steps == 0:
                        print('Advanced to step {:d}'.format(step))
                    if step == resume_step:
                        break
                if step == resume_step:
                    step += 1
                    break
                start_epoch += 1
        else:
            step = resume_step + 1

    # Make run directories
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(get_output_dir(args.run_dir), exist_ok=True)
    os.makedirs(get_checkpoint_dir(args.run_dir), exist_ok=True)

    start_time = time.time()
    for e in range(start_epoch, args.train_epochs):
        print('Starting epoch {:d}.'.format(e))
        # Train loop
        if args.profiler:
            prof = torch.autograd.profiler.profile().__enter__()
        for i_batch, (frames, cameras) in enumerate(train_dataloader):
            if args.profiler and i_batch == 5:
                break

            optimizer.zero_grad()

            batch_size = frames.shape[0]

            num_context_views = min(
                6, np.random.randint(1, frames.shape[1] - 1))
            num_query_views = min(
                3, np.random.randint(1, frames.shape[1] - num_context_views))

            b = np.random.random(frames.shape[0:2])
            idxs = np.argsort(b, axis=-1)
            input_idx = idxs[:, :num_context_views]
            query_idx = \
                idxs[:, num_context_views:num_context_views + num_query_views]
            all_idx = idxs[:, :num_context_views + num_query_views]
            t = np.arange(frames.shape[0])[:,None]

            all_images = frames[t, all_idx, ...]
            all_poses = cameras[t, all_idx, ...]
            input_images = frames[t, input_idx, ...]
            input_poses = cameras[t, input_idx, ...]
            query_images = frames[t, query_idx, ...]
            query_poses = cameras[t, query_idx, ...]

            if cuda:
                all_images = all_images.cuda()
                all_poses = all_poses.cuda()
                input_images = input_images.cuda()
                input_poses = input_poses.cuda()
                query_images = query_images.cuda()
                query_poses = query_poses.cuda()

            train_args = {
                'step': step,
                'dp_rep_net': dp_rep_net,
                'rep_net': rep_net,
                'renderer': renderer,
                'optimizer': optimizer,
                'input_images': input_images,
                'input_poses': input_poses,
                'query_images': query_images,
                'query_poses': query_poses,
                'all_images': all_images,
                'all_poses': all_poses,
                'writer': writer,
                'log_steps': args.log_steps,
                'checkpoint_steps': args.checkpoint_steps,
                'run_path': args.run_dir,
                
            }
            #gqn_train_iter(train_args)
            proj_train_iter(train_args)


        if args.profiler:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace("repro.prof")

        # Eval loop
    writer.close()



if __name__ == '__main__':
    # command line argument parser
    parser = argparse.ArgumentParser(
        description='Train a GQN using Pytorch.')
    # directory parameters
    parser.add_argument(
        '--data-dir', type=str, default='/tmp/data/gqn-dataset',
        help='The path to the gqn-dataset-pytorch directory.')
    parser.add_argument(
        '--dataset', type=str, default='rooms_ring_camera',
        help='The name of the GQN dataset to use. \
        Available names are: \
        scannet | jaco | mazes | rooms_free_camera_no_object_rotations | \
        rooms_free_camera_with_object_rotations | rooms_ring_camera | \
        shepard_metzler_5_parts | shepard_metzler_7_parts')
    parser.add_argument(
        '--run-dir', type=str, default='/tmp/models/gqn',
        help='The directory where the state of the training session will be stored.')
    parser.add_argument(
        '--resume-from', type=str,
        help='The checkpoint file to resume from.')
    # training parameters
    parser.add_argument(
        '--model', type=str, default='jump-vae', 
        help="The model type to use.")
    parser.add_argument(
        '--cuda', default=False, action='store_true',
        help="Run the model using CUDA.")
    parser.add_argument(
        '--data-parallel', default=False, action='store_true',
        help="Use data parallelism to execute the model.")
    parser.add_argument(
        '--train-epochs', type=int, default=40,
        help='The number of epochs to train.')
    parser.add_argument(
        '--batch-size', type=int, default=8,  # 36 reported in GQN paper -> multi-GPU?
        help='The number of data points per batch. One data point is a tuple of \
        ((query_camera_pose, [(context_frame, context_camera_pose)]), target_frame).')
    # snapshot parameters
    parser.add_argument(
        '--checkpoint-steps', type=int, default=5000,
        help='Number of steps between checkpoints.')
    # data loading
    parser.add_argument(
        '--queue-threads', type=int, default=-1,
        help='How many parallel threads to run for data queuing.')
    # logging
    parser.add_argument(
        '--log-steps', type=int, default=100,
        help='Global steps between log output.')
    parser.add_argument(
        '--debug', default=False, action='store_true',
        help="Enables debugging mode for more verbose logging.")
    parser.add_argument(
        '--initial-eval', default=False, action='store_true',
        help="Runs an evaluation before the first training iteration.")
    parser.add_argument(
        '--profiler', default=False, action='store_true',
        help="Run the profiler for 5 runs")

    args = parser.parse_args()

    train(args)
