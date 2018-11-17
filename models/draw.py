from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions.normal

class Generator(nn.Module):
    def __init__(self, input_shape, hidden_channels, z_channels, canvas_channels,
                 canvas_conv_size=4, lstm_conv_size=1, lstm_padding=0):
        super(Generator, self).__init__()

        self._input_shape = input_shape

        self._lstm_input_channels = input_shape[1]
        self._lstm_hidden_channels = hidden_channels
        self._z_channels = z_channels
        self._u_channels = canvas_channels
        self._canvas_conv_size = canvas_conv_size
        self._lstm_kernel_size = lstm_conv_size
        self._lstm_padding = lstm_padding

        # LSTM variables
        self.input_gate_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                         self._lstm_kernel_size, padding=self._lstm_padding)
        self.new_input_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                        self._lstm_kernel_size, padding=self._lstm_padding)
        self.forget_gate_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                          self._lstm_kernel_size, padding=self._lstm_padding)
        self.output_gate_conv = nn.Conv2d(self._lstm_input_channels, self._lstm_hidden_channels,
                                          self._lstm_kernel_size, padding=self._lstm_padding)

        self.u_deconv = nn.ConvTranspose2d(self._lstm_hidden_channels, self._u_channels, self._canvas_conv_size,
                                           stride=self._canvas_conv_size)

        # For sampling the latent z value
        self.prior_factor_conv = nn.Conv2d(self._lstm_hidden_channels, self._z_channels * 2,
                                           self._lstm_kernel_size, padding=self._lstm_padding)

    def init(self, batch=1):
        # Hidden, state, canvas
        return (
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]),
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), 
            torch.zeros(batch, self._u_channels,
                        self._input_shape[2] * self._canvas_conv_size,
                        self._input_shape[3] * self._canvas_conv_size))

    def prior_distribution(self, hidden):
        prior_factor = self.prior_factor_conv(hidden)
        mu, sigma = torch.split(prior_factor, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, F.softplus(sigma))
        return d

    def forward(self, inputs, z, hidden, state, u):
        # Concatenate the input and z vector
        inputs = torch.cat((inputs, z), 1)

        fg = torch.sigmoid(self.forget_gate_conv(inputs))
        ig = torch.sigmoid(self.input_gate_conv(inputs))
        ni = torch.tanh(self.new_input_conv(inputs))
        og = torch.sigmoid(self.output_gate_conv(inputs))

        new_state = (state * fg) + (ig * ni)
        new_hidden = torch.tanh(new_state) * og

        new_u = u + self.u_deconv(new_hidden)

        return new_hidden, new_state, new_u


class Inference(nn.Module):
    def __init__(self, input_shape, hidden_channels, z_channels, canvas_channels,
                 canvas_conv_size=4, lstm_conv_size=1, lstm_padding=0):
        super(Inference, self).__init__()

        self._input_shape = input_shape

        self._z_channels = z_channels
        self._lstm_hidden_channels = hidden_channels
        self._lstm_kernel_size = lstm_conv_size
        self._lstm_padding = lstm_padding
        self._u_channels = canvas_channels
        self._canvas_conv_size = canvas_channels

        self._lstm_input_channels = input_shape[1]

        lstm_channels = input_shape[1]
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
        # Hidden, state
        return torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3])

    def posterior_distribution(self, h):
        posterior_factor = self.posterior_factor_conv(h)
        mu, sigma = torch.split(
            posterior_factor, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, F.softplus(sigma))
        return d

    def forward(self, inputs, hidden, state):
        # Perform ConvLSTM gate computation
        fg = torch.sigmoid(self.forget_gate_conv(inputs))
        ig = torch.sigmoid(self.input_gate_conv(inputs))
        ni = torch.tanh(self.new_input_conv(inputs))
        og = torch.sigmoid(self.output_gate_conv(inputs))

        # Update lstm state
        new_state = (state * fg) + (ig * ni)
        new_hidden = torch.tanh(new_state) * og

        return new_hidden, new_state
