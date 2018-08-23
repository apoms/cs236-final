from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F

Z_CHANNELS = 64
LSTM_CHANNELS = 256
U_CHANNELS = 256

class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()

        self._input_shape = input_shape
        
        pose_channels = 7
        representation_channels = input_shape[1]
        z_channels = Z_CHANNELS
        self._lstm_hidden_channels = LSTM_CHANNELS
        self._lstm_input_channels = (
            self._lstm_hidden_channels + pose_channels + representation_channels + z_channels)
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

    def init(self, batch=1):
        return torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._lstm_hidden_channels, self._input_shape[2], self._input_shape[3]), \
            torch.zeros(batch, self._u_channels, self._input_shape[2] * 4, self._input_shape[3] * 4)

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
