from __future__ import print_function

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from gqn_generator import Generator, Inference
from gqn_encoder import Tower
from datasets.gqn_dataset import GQNDataset
from datasets.scannet_dataset import ScanNetDataset
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


LSTM_LAYERS = 8

PIXEL_ANNEALING_ITERS = 2e5
PIXEL_SIGMA_INITIAL = 2.0
PIXEL_SIGMA_FINAL = 0.7

LR_ANNEALING_ITERS = 1.6e6
LR_INITIAL = 5e-4
LR_FINAL = 5e-5


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


def decode_inference_rnn(
        generator,
        inference,
        representation,
        query_image,
        query_pose,
        lstm_layers,
        cuda=False,
        debug=False):
    '''
    Perform inference of the query_image using the inference and generator
    networks.
    '''

    if isinstance(generator, nn.DataParallel):
        orig_generator = generator.module
    else:
        orig_generator = generator

    if isinstance(inference, nn.DataParallel):
        orig_inference = inference.module
    else:
        orig_inference = inference

    batch_size = query_image.shape[0]
    hidden_g, state_g, u_g = orig_generator.init(batch_size)
    hidden_i, state_i = orig_inference.init(batch_size)

    if cuda:
        hidden_g = hidden_g.cuda()
        state_g = state_g.cuda()
        u_g = u_g.cuda()
        hidden_i = hidden_i.cuda()
        state_i = state_i.cuda()

    inference_time = 0
    generator_time = 0
    kl_divs = []
    for layer in range(lstm_layers):
        # Prior distribution
        d_g = orig_generator.prior_distribution(hidden_g)

        # Update inference lstm state
        s = time.time()
        hidden_i, state_i = inference(
            query_image, query_pose, representation, hidden_i, state_i,
            hidden_g, u_g)
        inference_time += (time.time() - s)

        # Posterior distribution
        d_i = orig_inference.posterior_distribution(hidden_i)

        # Posterior sample
        z_i = d_i.rsample()

        # Update generator lstm state
        s = time.time()
        hidden_g, state_g, u_g = generator(query_pose, representation, z_i,
                                           hidden_g, state_g, u_g)
        generator_time += (time.time() - s)

        # ELBO KL loss
        kl_div = torch.distributions.kl.kl_divergence(d_i, d_g)
        kl_divs.append(kl_div)

    if debug:
        print('Inference time: {:.2f}'.format(inference_time))
        print('Generator time: {:.2f}'.format(generator_time))

    return u_g, kl_divs


def decode_generator_rnn(generator, representation, query_pose, lstm_layers,
                         cuda=False):
    '''
    Perform inference of the query_image using the generator network.
    '''

    if isinstance(generator, nn.DataParallel):
        orig_generator = generator.module
    else:
        orig_generator = generator

    batch_size = query_pose.shape[0]
    hidden_g, state_g, u_g = generator.init(batch_size)

    if cuda:
        hidden_g = hidden_g.cuda()
        state_g = state_g.cuda()
        u_g = u_g.cuda()

    kl_divs = []
    for layer in range(lstm_layers):
        # Prior distribution
        d_g = orig_generator.prior_distribution(hidden_g)

        # Sample from the prior
        z_g = d_g.sample()

        # Update generator lstm state
        hidden_g, state_g, u_g = generator(query_pose, representation, z_g,
                                           hidden_g, state_g, u_g)
    return u_g


def train(args):
    cuda = args.cuda
    BATCH_SIZE = args.batch_size

    writer = tensorboardX.SummaryWriter(
        log_dir=os.path.join(args.run_dir, 'tensorboard'))

    if args.dataset == 'scannet':
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

    rep_net = Tower(pose_channels)

    # Pass dummy data through the network to get the representation
    # shape for the Generator and Inference networks
    dummy_images = torch.rand(1, 3, 64, 64)
    dummy_poses = torch.rand(1, pose_channels, 1, 1)
    dummy_rep = rep_net(dummy_images, dummy_poses)

    generator = Generator(dummy_rep.shape, pose_channels)
    inference = Inference(dummy_rep.shape, pose_channels)

    if args.data_parallel:
        dp_rep_net = nn.DataParallel(rep_net)
        dp_generator = nn.DataParallel(generator)
        dp_inference = nn.DataParallel(inference)
    else:
        dp_rep_net = rep_net
        dp_generator = generator
        dp_inference = inference

    if cuda:
        dp_rep_net = dp_rep_net.cuda()
        dp_generator = dp_generator.cuda()
        dp_inference = dp_inference.cuda()

    model_params = (
        list(dp_rep_net.parameters()) +
        list(dp_generator.parameters()) +
        list(dp_inference.parameters()))
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
        generator.load_state_dict(checkpoint['generator_net'])
        inference.load_state_dict(checkpoint['inference_net'])
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

            num_context_views = np.random.randint(1, frames.shape[1])
            b = np.random.random(frames.shape[0:2])
            idxs = np.argsort(b, axis=-1)
            input_idx = idxs[:, :num_context_views]
            query_idx = idxs[:, num_context_views:num_context_views+1]
            t = np.arange(frames.shape[0])[:,None]

            input_images = frames[t, input_idx, ...]
            input_poses = cameras[t, input_idx, ...]
            query_image = frames[t, query_idx, ...]
            query_pose = cameras[t, query_idx, ...]

            query_image = torch.squeeze(query_image, dim=1)
            query_pose = torch.squeeze(query_pose, dim=1)

            if cuda:
                input_images = input_images.cuda()
                input_poses = input_poses.cuda()
                query_image = query_image.cuda()
                query_pose = query_pose.cuda()

            encode_start = time.time()
            rep = encode_representation(dp_rep_net, input_images, input_poses)
            encode_end = time.time()

            decode_start = time.time()
            u_g, kl_divs = decode_inference_rnn(
                generator=dp_generator,
                inference=dp_inference,
                representation=rep,
                query_image=query_image,
                query_pose=query_pose,
                lstm_layers=LSTM_LAYERS,
                cuda=cuda,
                debug=args.debug)
            decode_end = time.time()

            ELBO = 0

            kl = kl_divs[0]
            for i in range(1, len(kl_divs)):
                kl += kl_divs[i]
            ELBO += torch.sum(torch.mean(kl, dim=0))

            KL_loss = ELBO.cpu().data

            # Observation distribution
            observation_sigma = calculate_pixel_sigma(step)
            d_obs = generator.observation_distribution(u_g, observation_sigma)
            # ELBO reconstruction loss
            log_likelihood = torch.sum(
                torch.mean(d_obs.log_prob(query_image), dim=0))
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
                print('Decode time: {:.2f}'.format(decode_end - decode_start))
                print('Backward time: {:.2f}'.format(backward_end - backward_start))
                print()

            elapsed_time = time.time() - start_time
            formatted_time = str(datetime.timedelta(seconds=elapsed_time))
            if step % args.log_steps == 0:
                print('Logging step {:d} (epoch {:d}, {:d}/{:d})...'.format(
                    step, e, i_batch, len(train_dataloader)))
                print('Elapsed time: {:s}'.format(formatted_time))
                print('KL Loss: ', KL_loss)
                print('LL Loss: ', log_likelihood.cpu().data)
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
                print(img.shape)
                path = get_output_dir(args.run_dir)
                scipy.misc.imsave(
                    os.path.join(path, 'sample{:06d}.png'.format(step)), img)
                print('Done logging.')
                writer.add_image(
                    'data/obs_image',
                    torchvision.utils.make_grid(d_obs.mean, normalize=True, scale_each=True),
                    step)
                writer.add_image(
                    'data/target_image', torchvision.utils.make_grid(query_image), step)
                writer.add_scalar('data/kl_loss', KL_loss, step)
                writer.add_scalar('data/nll_loss', -log_likelihood.cpu().data, step)
                writer.add_scalar('data/loss', loss, step)
                writer.add_scalar('param/sigma', observation_sigma, step)
                writer.add_scalar('param/lr', new_lr, step)
                print()

            if step % args.checkpoint_steps == 0:
                print('Checkpointing step {:d}...'.format(step))
                print('Elapsed time: {:s}'.format(formatted_time))
                path = get_checkpoint_path(args.run_dir, step)
                data = {
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'representation_net': rep_net.state_dict(),
                    'generator_net': generator.state_dict(),
                    'inference_net': inference.state_dict(),
                    'dataset': args.dataset,
                    'seed': seed,
                    'total_time': elapsed_time,
                }
                write_checkpoint(path, data)
                print('Done checkpointing.')
                print()

            step += 1

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
        '--cuda', default=False, action='store_true',
        help="Run the model using CUDA.")
    parser.add_argument(
        '--data-parallel', default=False, action='store_true',
        help="Use data parallelism to execute the model.")
    parser.add_argument(
        '--train-epochs', type=int, default=40,
        help='The number of epochs to train.')
    parser.add_argument(
        '--batch-size', type=int, default=36,  # 36 reported in GQN paper -> multi-GPU?
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
