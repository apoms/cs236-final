from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
import models.draw as draw
import torch.distributions.normal
import time

GLOBAL_Z_CHANNELS = 128
Z_CHANNELS = 64
LSTM_CHANNELS = 256
U_CHANNELS = 256


class VAEFunctionGenerator(nn.Module):
    def __init__(self, input_shape, model_settings={}, train=True):
        super(VAEFunctionGenerator, self).__init__()

        self._input_shape = input_shape
        self._pose_channels = pose_channels
        self._representation_channels = input_shape[1]
        self._train = train

        def opt(param, default):
            return model_settings[param] if param in model_settings else default

        self._lstm_layers = opt('lstm_layers', 8)
        self._z_channels = opt('z_channels', 64)
        self._lstm_hidden_channels = opt('lstm_hidden_channels', 128)
        self._canvas_channels = opt('canvas_channels', 256)
        self._canvas_conv_size = opt('canvas_conv_size', 4)


        self._input_channels = representation_channels
        self._z_channels = GLOBAL_Z_CHANNELS
        self._kernel_size = 5
        self._padding = 2

        self.global_dist_conv = nn.Conv2d(self._input_channels,
                                          self._z_channels * 2,
                                          self._kernel_size, padding=self._padding)

    def z_distribution(self, representation):
        '''Outputs the distribution for the global latent variable Z in Neural Processes'''
        global_dist = self.global_dist_conv(representation)
        mu, sigma = torch.split(global_dist, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, 0.1 + 0.9 * F.softplus(sigma))
        return d

    def forward(self, representation, target_representation=None):
        '''Outputs the distribution for the global latent variable Z in Neural Processes'''
        input_dist = self.z_distribution(representation)

        if self._train:
            target_dist = self.z_distribution(target_representation)
            kl_z = torch.distributions.kl.kl_divergence(target_dist, input_dist)
            return target_dist, [kl_z]
        else:
            return input_dist, []


class DRAWFunctionGenerator(nn.Module):
    def __init__(self, input_shape, model_settings={}, train=True):
        super(DRAWFunctionGenerator, self).__init__()

        self._input_shape = input_shape
        self._representation_channels = input_shape[1]
        self._train = train

        def opt(param, default):
            return model_settings[param] if param in model_settings else default

        self._lstm_layers = opt('lstm_layers', 6)
        self._z_channels = opt('z_channels', 3)
        self._lstm_hidden_channels = opt('lstm_hidden_channels', 128)
        self._lstm_conv_size = opt('lstm_conv_size', 3)
        self._lstm_padding = opt('lstm_padding', 1)
        self._canvas_channels = opt('canvas_channels', 256)
        self._canvas_conv_size = opt('canvas_conv_size', 2)

        self._generator_input_channels = (
            self._lstm_hidden_channels +
            self._lstm_hidden_channels +
            self._z_channels)
        self._generator_input_shape = (input_shape[0], self._generator_input_channels,
                                       input_shape[2] // self._canvas_conv_size,
                                       input_shape[3] // self._canvas_conv_size)
        self.generator = draw.Generator(
            input_shape=self._generator_input_shape,
            hidden_channels=self._lstm_hidden_channels,
            z_channels=self._z_channels,
            canvas_channels=self._canvas_channels,
            canvas_conv_size=self._canvas_conv_size,
            lstm_conv_size=self._lstm_conv_size,
            lstm_padding=self._lstm_padding
        )

        # Downsampling convs
        self.representation_downsample_conv = nn.Conv2d(self._representation_channels,
                                                        self._lstm_hidden_channels,
                                                        self._canvas_conv_size,
                                                        stride=self._canvas_conv_size)
        self.u_g_downsample_conv = nn.Conv2d(self._canvas_channels,
                                             self._lstm_hidden_channels,
                                             self._canvas_conv_size,
                                             stride=self._canvas_conv_size)

        self._inference_input_channels = (
            # hidden, rep_target, rep_input, ncanvas, hidden_g
            self._lstm_hidden_channels * 5
            )

        # Bottleneck conv
        self._bottleneck_channels = 128
        self.input_bottleneck_conv = nn.Conv2d(self._inference_input_channels, self._bottleneck_channels, 1)

        self._inference_input_shape = (input_shape[0], self._bottleneck_channels,
                                       input_shape[2] // self._canvas_conv_size,
                                       input_shape[3] // self._canvas_conv_size)
        self.inference = draw.Inference(
            input_shape=self._inference_input_shape,
            hidden_channels=self._lstm_hidden_channels,
            z_channels=self._z_channels,
            canvas_channels=self._canvas_channels,
            canvas_conv_size=self._canvas_conv_size,
            lstm_conv_size=self._lstm_conv_size,
            lstm_padding=self._lstm_padding
        )

    def _generator(self, representation, z_i, hidden_g, state_g, u_g):
        # Concatenate inputs for generator
        downsampled_rep = self.representation_downsample_conv(representation)
        inputs = torch.cat((hidden_g, downsampled_rep), 1)
        return self.generator(inputs, z_i, hidden_g, state_g, u_g)

    def _inference(self, target_representation, input_representation, hidden, state, hidden_g, u_g):
        # Transform the image and u_g to the hidden vector size
        downsampled_target = self.representation_downsample_conv(target_representation)
        downsampled_input = self.representation_downsample_conv(input_representation)
        downsampled_u_g = self.u_g_downsample_conv(u_g)
        # Concatenate the hidden vector, pose, representation, image, u_g,
        # and hidden_g vector
        inputs = torch.cat((hidden, downsampled_target, downsampled_input,
                            downsampled_u_g, hidden_g), 1)
        bn_inputs = self.input_bottleneck_conv(inputs)
        return self.inference(bn_inputs, hidden, state)

    def _train_forward(self, input_representation, target_representation, debug=False):
        '''
        Perform inference of the query_image using the inference and generator
        networks.
        '''
        device = input_representation.device
        batch_size = input_representation.shape[0]
        hidden_g, state_g, u_g = self.generator.init(batch_size)
        hidden_i, state_i = self.inference.init(batch_size)

        hidden_g = hidden_g.to(device=device)
        state_g = state_g.to(device=device)
        u_g = u_g.to(device=device)
        hidden_i = hidden_i.to(device=device)
        state_i = state_i.to(device=device)

        inference_time = 0
        generator_time = 0
        kl_divs = []
        for layer in range(self._lstm_layers):
            # Prior distribution
            d_g = self.generator.prior_distribution(hidden_g)

            # Update inference lstm state
            s = time.time()
            hidden_i, state_i = self._inference(target_representation, input_representation,
                                                hidden_i, state_i, hidden_g, u_g)
            inference_time += (time.time() - s)

            # Posterior distribution
            d_i = self.inference.posterior_distribution(hidden_i)

            # Posterior sample
            z_i = d_i.rsample()

            # Update generator lstm state
            s = time.time()
            hidden_g, state_g, u_g = self._generator(input_representation, z_i,
                                                     hidden_g, state_g, u_g)
            generator_time += (time.time() - s)

            # ELBO KL loss
            kl_div = torch.distributions.kl.kl_divergence(d_i, d_g)
            kl_divs.append(kl_div)

        if debug:
            print('Inference time: {:.2f}'.format(inference_time))
            print('Generator time: {:.2f}'.format(generator_time))

        return u_g, kl_divs

    def _test_forward(self, representation, debug=False):
        '''
        Perform inference of the query_image using the generator network.
        '''
        device = representation.device

        batch_size = representation.shape[0]
        hidden_g, state_g, u_g = self.generator.init(batch_size)

        hidden_g = hidden_g.to(device=device)
        state_g = state_g.to(device=device)
        u_g = u_g.to(device=device)

        kl_divs = []
        for layer in range(self._lstm_layers):
            # Prior distribution
            d_g = self.generator.prior_distribution(hidden_g)

            # Sample from the prior
            z_g = d_g.sample()

            # Update generator lstm state
            hidden_g, state_g, u_g = self._generator(representation, z_g,
                                                     hidden_g, state_g, u_g)
        return u_g

    def forward(self, representation, target_representation=None):
        '''Outputs the distribution for the global latent variable Z in Neural Processes'''
        # Run the draw network to generate the canvas
        if self._train:
            u_g, kl_divs = self._train_forward(representation, target_representation)
        else:
            u_g = self._test_forward(representation)
            kl_divs = []

        # Convert canvas into normal distribution
        mu, sigma = torch.split(u_g, split_size_or_sections=self._canvas_channels // 2, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, F.softplus(sigma))
        return d, kl_divs


class Generator(nn.Module):
    def __init__(self, input_shape, pose_channels=7):
        super(Generator, self).__init__()

        self._input_shape = input_shape

        image_channels = 3
        self._z_channels = GLOBAL_Z_CHANNELS
        self._lstm_hidden_channels = LSTM_CHANNELS
        self._lstm_input_channels = (
            self._lstm_hidden_channels + pose_channels + self._z_channels)
        self._lstm_kernel_size = 5
        self._lstm_padding = 2
        self._u_channels = U_CHANNELS

        # LSTM variables
        self.input_gate_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                         self._lstm_kernel_size, padding=self._lstm_padding)
        self.new_input_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                        self._lstm_kernel_size, padding=self._lstm_padding)
        self.forget_gate_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                          self._lstm_kernel_size, padding=self._lstm_padding)
        self.output_gate_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                          self._lstm_kernel_size, padding=self._lstm_padding)

        self.u_deconv = nn.ConvTranspose2d(self._lstm_hidden_channels, self._u_channels, 4, stride=4)

        # For sampling the query image
        self.observation_factor_conv = nn.Conv2d(self._u_channels, image_channels,
                                           self._lstm_kernel_size, padding=self._lstm_padding)

    def init(self, batch=1):
        # Hidden, state, and u
        return torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._u_channels, self._input_shape[2] * 4, self._input_shape[3] * 4)

    def observation_distribution(self, u, sigma):
        mu = self.observation_factor_conv(u)
        d = torch.distributions.normal.Normal(mu, sigma)
        return d

    def forward(self, pose, z, hidden, state, u):
        # Concatenate the pose, representation, and z vector
        pose_e = pose.expand(-1, -1, z.shape[2], z.shape[3])
        inputs = torch.cat((hidden, pose_e, z), 1)

        fg = torch.sigmoid(self.forget_gate_conv(inputs))
        ig = torch.sigmoid(self.input_gate_conv(inputs))
        ni = torch.tanh(self.new_input_conv(inputs))
        og = torch.sigmoid(self.output_gate_conv(inputs))

        new_state = (state * fg) + (ig * ni)
        new_hidden = torch.tanh(new_state) * og

        new_u = u + self.u_deconv(new_hidden)

        return new_hidden, new_state, new_u


class Renderer(nn.Module):
    def __init__(self, input_shape, pose_channels=7, model_settings={}, train=True):
        super(Renderer, self).__init__()

        self._query_image_channels = 3

        self._input_shape = input_shape
        self._pose_channels = pose_channels
        self._representation_channels = input_shape[1]
        self._lstm_layers = 6
        self._train = train

        fn_generator_type = model_settings['fn_generator_type']
        if fn_generator_type == 'DRAW':
            self.fn_generator = DRAWFunctionGenerator(input_shape, model_settings, train=train)
        elif fn_generator_type == 'VAE':
            self.fn_generator = VAEFunctionGenerator(input_shape, model_settings, train=train)
        else:
            print('Unknown model type for function generator: {:s}'.format(
                fn_generator_type))
            exit(-1)

        self.generator = Generator(input_shape, pose_channels)

    def observation_distribution(self, u, sigma):
        return self.generator.observation_distribution(u, sigma)

    def z_distribution_with_loss(self, representation, target_representation=None):
        return self.fn_generator(representation, target_representation)

    def z_distribution(self, representation, target_representation=None):
        return self.fn_generator(representation, target_representation)[0]

    def sample_z(self, representation, target_representation=None):
        return self.z_distribution(representation, target_representation)[0].rsample()

    def forward(self, global_z, query_pose, query_image=None, cuda=False, debug=False):
        '''
        Perform inference of the query_image using the generator network.
        '''
        batch_size = query_pose.shape[0]
        hidden_g, state_g, u_g = self.generator.init(batch_size)

        if cuda:
            hidden_g = hidden_g.cuda()
            state_g = state_g.cuda()
            u_g = u_g.cuda()

        for layer in range(self._lstm_layers):
            # Update generator lstm state
            hidden_g, state_g, u_g = self.generator(query_pose, global_z, hidden_g, state_g, u_g)

        return u_g
