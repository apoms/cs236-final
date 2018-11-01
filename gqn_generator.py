from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions.normal

Z_CHANNELS = 64
LSTM_CHANNELS = 256
U_CHANNELS = 256

class FunctionGenerator(nn.Module):
    def __init__(self, input_shape):
        super(FunctionGenerator, self).__init__()

        representation_channels = input_shape[1]

        self._input_channels = representation_channels
        self._z_channels = Z_CHANNELS
        self._kernel_size = 5
        self._padding = 2

        self.global_dist_conv = nn.Conv2d(self._input_channels,
                                          self._z_channels * 2,
                                          self._kernel_size, padding=self._padding)

    def forward(self, representation):
        '''Outputs the distribution for the global latent variable Z in Neural Processes'''
        global_dist = self.global_dist_conv(representation)
        mu, sigma = torch.split(global_dist, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, F.softplus(sigma))
        return d


class Generator(nn.Module):
    def __init__(self, input_shape, pose_channels=7):
        super(Generator, self).__init__()

        self._input_shape = input_shape

        image_channels = 3
        representation_channels = input_shape[1]
        self._z_channels = Z_CHANNELS
        self._lstm_hidden_channels = LSTM_CHANNELS
        self._lstm_input_channels = (
            self._lstm_hidden_channels + pose_channels + representation_channels + self._z_channels)
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

        # For sampling the latent z value
        self.prior_factor_conv = nn.Conv2d(self._lstm_hidden_channels, self._z_channels * 2,
                                           self._lstm_kernel_size, padding=self._lstm_padding)

        # For sampling the query image
        self.observation_factor_conv = nn.Conv2d(self._u_channels, image_channels,
                                           self._lstm_kernel_size, padding=self._lstm_padding)

    def init(self, batch=1):
        return torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._u_channels, self._input_shape[2] * 4, self._input_shape[3] * 4)

    def prior_distribution(self, h):
        prior_factor = self.prior_factor_conv(h)
        mu, sigma = torch.split(prior_factor, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, F.softplus(sigma))
        return d

    def observation_distribution(self, u, sigma):
        mu = self.observation_factor_conv(u)
        d = torch.distributions.normal.Normal(mu, sigma)
        return d

    def forward(self, pose, representation, z, hidden, state, u):
        # Concatenate the pose, representation, and z vector
        pose_e = pose.expand(-1, -1, representation.shape[2], representation.shape[3])
        inputs = torch.cat((hidden, pose_e, representation, z), 1)

        fg = torch.sigmoid(self.forget_gate_conv(inputs))
        ig = torch.sigmoid(self.input_gate_conv(inputs))
        ni = torch.tanh(self.new_input_conv(inputs))
        og = torch.sigmoid(self.output_gate_conv(inputs))

        new_state = (state * fg) + (ig * ni)
        new_hidden = torch.tanh(new_state) * og

        new_u = u + self.u_deconv(new_hidden)

        return new_hidden, new_state, new_u


class Inference(nn.Module):
    def __init__(self, input_shape, pose_channels=7):
        super(Inference, self).__init__()

        self._input_shape = input_shape

        image_channels = 3
        representation_channels = input_shape[1]
        self._z_channels = Z_CHANNELS
        self._lstm_hidden_channels = LSTM_CHANNELS
        self._lstm_kernel_size = 5
        self._lstm_padding = 2
        self._u_channels = U_CHANNELS

        self._lstm_input_channels = (
            self._lstm_hidden_channels * 2 + pose_channels + representation_channels +
            self._u_channels + image_channels * 4)

        self._bottleneck_channels = 128

        # Downsampling convs
        self.image_downsample_conv = nn.Conv2d(image_channels, image_channels * 4, 4, stride=4)
        self.u_g_downsample_conv = nn.Conv2d(self._u_channels, self._u_channels, 4, stride=4)

        # Bottleneck conv
        self.input_bottleneck_conv = nn.Conv2d(self._lstm_input_channels, self._bottleneck_channels, 1)

        lstm_channels = self._bottleneck_channels
        # LSTM variables
        self.input_gate_conv = nn.Conv2d(lstm_channels, self._lstm_hidden_channels,
                                         self._lstm_kernel_size, padding=self._lstm_padding)
        self.new_input_conv = nn.Conv2d(lstm_channels, self._lstm_hidden_channels,
                                        self._lstm_kernel_size, padding=self._lstm_padding)
        self.forget_gate_conv = nn.Conv2d(lstm_channels, self._lstm_hidden_channels,
                                          self._lstm_kernel_size, padding=self._lstm_padding)
        self.output_gate_conv = nn.Conv2d(lstm_channels, self._lstm_hidden_channels,
                                          self._lstm_kernel_size, padding=self._lstm_padding)

        # For sampling the latent z value
        self.posterior_factor_conv = nn.Conv2d(self._lstm_hidden_channels, self._z_channels * 2,
                                          self._lstm_kernel_size, padding=self._lstm_padding)

    def init(self, batch=1):
        return torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3])

    def posterior_distribution(self, h):
        posterior_factor = self.posterior_factor_conv(h)
        mu, sigma = torch.split(
            posterior_factor, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, F.softplus(sigma))
        return d

    def forward(self, image, pose, representation, hidden, state, hidden_g, u_g):
        # Transform the image and u_g to the hidden vector size
        downsampled_image = self.image_downsample_conv(image)
        downsampled_u_g = self.u_g_downsample_conv(u_g)
        # Concatenate the hidden vector, pose, representation, image, u_g,
        # and hidden_g vector
        pose_e = pose.expand(
            -1, -1, representation.shape[2], representation.shape[3])
        inputs = torch.cat((hidden, pose_e, representation, downsampled_image,
                            downsampled_u_g, hidden_g), 1)

        bn_inputs = self.input_bottleneck_conv(inputs)

        # Perform ConvLSTM gate computation
        fg = torch.sigmoid(self.forget_gate_conv(bn_inputs))
        ig = torch.sigmoid(self.input_gate_conv(bn_inputs))
        ni = torch.tanh(self.new_input_conv(bn_inputs))
        og = torch.sigmoid(self.output_gate_conv(bn_inputs))

        # Update lstm state
        new_state = (state * fg) + (ig * ni)
        new_hidden = torch.tanh(new_state) * og

        return new_hidden, new_state
