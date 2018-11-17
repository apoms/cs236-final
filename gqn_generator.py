from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
from models.draw import Generator, Inference
import torch.distributions.normal

Z_CHANNELS = 64
LSTM_CHANNELS = 256
U_CHANNELS = 256


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


class Renderer(nn.Module):
    def __init__(self, input_shape, pose_channels=7, model_settings={}, train=True):
        super(Renderer, self).__init__()

        # RGB Image
        self._query_image_channels = 3

        self._input_shape = input_shape
        self._pose_channels = pose_channels
        self._representation_channels = input_shape[1]
        self._train = train

        def opt(param, default):
            return model_settings[param] if param in model_settings else default

        self._lstm_layers = opt('lstm_layers', 8)
        self._z_channels = opt('z_channels', 64)
        self._lstm_hidden_channels = opt('lstm_hidden_channels', 256)
        self._lstm_conv_size = opt('lstm_conv_size', 5)
        self._lstm_padding = opt('lstm_padding', 2)
        self._canvas_channels = opt('canvas_channels', 256)
        self._canvas_conv_size = opt('canvas_conv_size', 4)

        self._generator_input_channels = (
            self._lstm_hidden_channels +
            self._pose_channels +
            self._representation_channels +
            self._z_channels)
        self._generator_input_shape = (input_shape[0], self._generator_input_channels, input_shape[2], input_shape[3])
        self.generator = Generator(
            input_shape=self._generator_input_shape
            hidden_channels=self._lstm_hidden_channels,
            z_channels=self._z_channels,
            canvas_channels=self._canvas_channels,
            canvas_conv_size=self._canvas_conv_size,
            lstm_conv_size=self._lstm_conv_size,
            lstm_padding=self._lstm_padding
        )

        # Downsampling convs
        self.image_downsample_conv = nn.Conv2d(image_channels, image_channels * 4, 4, stride=4)
        self.u_g_downsample_conv = nn.Conv2d(self._u_channels, self._u_channels, 4, stride=4)

        self._inference_input_channels = (
            self._lstm_hidden_channels * 2 +
            self._pose_channels +
            self._representation_channels +
            self._u_channels +
            self._query_image_channels * 4)

        # Bottleneck conv
        self._bottleneck_channels = 128
        self.input_bottleneck_conv = nn.Conv2d(self._inference_input_channels, self._bottleneck_channels, 1)

        self._inference_input_shape = (input_shape[0], self._bottleneck_channels, input_shape[2], input_shape[3])
        self.inference = Inference(self._inference_input_shape, pose_channels)

        # For sampling the query image
        self.observation_factor_conv = nn.Conv2d(self._u_channels, image_channels,
                                           self._lstm_kernel_size, padding=self._lstm_padding)

    def observation_distribution(self, u, sigma):
        mu = self.observation_factor_conv(u)
        d = torch.distributions.normal.Normal(mu, sigma)
        return d

    def _generator(self, query_pose, representation, z_i, hidden_g, state_g, u_g):
        # Concatenate inputs for generator
        pose_e = query_pose.expand(-1, -1, representation.shape[2], representation.shape[3])
        inputs = torch.cat((hidden_g, pose_e, representation), 1)
        return self.generator(inputs, z_i, hidden_g, state_g, u_g)

    def _inference(self, query_image, pose, representation, hidden, state, hidden_g, u_g):
        # Transform the image and u_g to the hidden vector size
        downsampled_image = self.image_downsample_conv(query_image)
        downsampled_u_g = self.u_g_downsample_conv(u_g)
        # Concatenate the hidden vector, pose, representation, image, u_g,
        # and hidden_g vector
        pose_e = pose.expand(
            -1, -1, representation.shape[2], representation.shape[3])
        inputs = torch.cat((hidden, pose_e, representation, downsampled_image,
                            downsampled_u_g, hidden_g), 1)
        bn_inputs = self.input_bottleneck_conv(inputs)
        return self.inference(bn_inputs, hidden, state)

    def _train_forward(representation, query_pose, query_image, cuda=False, debug=False):
        '''
        Perform inference of the query_image using the inference and generator
        networks.
        '''
        device = input_representation.device

        batch_size = query_image.shape[0]
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
        for layer in range(lstm_layers):
            # Prior distribution
            d_g = self.generator.prior_distribution(hidden_g)

            # Update inference lstm state
            s = time.time()
            hidden_i, state_i = self._inference(query_image, query_pose, representation,
                                                hidden_i, state_i, hidden_g, u_g)
            inference_time += (time.time() - s)

            # Posterior distribution
            d_i = self.inference.posterior_distribution(hidden_i)

            # Posterior sample
            z_i = d_i.rsample()

            # Update generator lstm state
            s = time.time()
            hidden_g, state_g, u_g = self._generator(query_pose, representation, z_i,
                                                     hidden_g, state_g, u_g)
            generator_time += (time.time() - s)

            # ELBO KL loss
            kl_div = torch.distributions.kl.kl_divergence(d_i, d_g)
            kl_divs.append(kl_div)

        if debug:
            print('Inference time: {:.2f}'.format(inference_time))
            print('Generator time: {:.2f}'.format(generator_time))

        return u_g, kl_divs

    def _test_forward(representation, query_pose, cuda=False, debug=False):
        '''
        Perform inference of the query_image using the generator network.
        '''

        batch_size = query_pose.shape[0]
        hidden_g, state_g, u_g = self.generator.init(batch_size)

        if cuda:
            hidden_g = hidden_g.cuda()
            state_g = state_g.cuda()
            u_g = u_g.cuda()

        kl_divs = []
        for layer in range(lstm_layers):
            # Prior distribution
            d_g = self.generator.prior_distribution(hidden_g)

            # Sample from the prior
            z_g = d_g.sample()

            # Update generator lstm state
            hidden_g, state_g, u_g = self.generator(query_pose, representation, z_g,
                                                    hidden_g, state_g, u_g)
        return u_g

    def forward(representation, query_pose, query_image=None,cuda=False, debug=False):
        if self._train:
            return self._train_forward(representation, query_pose, query_image, cuda, debug)
        else:
            return self._test_forward(representation, query_pose, cuda, debug)
