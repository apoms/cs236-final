from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import models.draw as draw
import torch.distributions.normal
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D


# Projection Net plan
#
# Idea:
# Context Input: (RGB image, Camera pose)
# Processing Context:
#   
# 

GLOBAL_Z_CHANNELS = 128
Z_CHANNELS = 64
LSTM_CHANNELS = 256
U_CHANNELS = 256


FLOAT_EPS = np.finfo(np.float).eps


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return w, x, y, z


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

# Taken from https://sscc.nimh.nih.gov/pub/dist/bin/linux_gcc32/meica.libs/nibabel/quaternions.py
def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY, 0.0 ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX, 0.0 ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY), 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]])


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
        '''Outputs the distribution for the global latent variable Z in Neural
        Processes'''
        global_dist = self.global_dist_conv(representation)
        mu, sigma = torch.split(global_dist, split_size_or_sections=self._z_channels, dim=1)
        # Softplus to constrain the sigma to be non-negative
        d = torch.distributions.normal.Normal(mu, 0.1 + 0.9 * F.softplus(sigma))
        return d

    def forward(self, representation, target_representation=None):
        '''Outputs the distribution for the global latent variable Z in Neural
        Processes'''
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
        self._generator_input_shape = (
            input_shape[0], self._generator_input_channels,
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
        self.representation_downsample_conv = nn.Conv2d(
            self._representation_channels,
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
        self.input_bottleneck_conv = nn.Conv2d(
            self._inference_input_channels, self._bottleneck_channels, 1)

        self._inference_input_shape = (
            input_shape[0], self._bottleneck_channels,
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

    def _inference(self, target_representation, input_representation, hidden,
                   state, hidden_g, u_g):
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

    def _train_forward(self, input_representation, target_representation,
                       debug=False):
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
        self.input_gate_conv = nn.Conv2d(
            self._lstm_input_channels, self._lstm_hidden_channels,
            self._lstm_kernel_size, padding=self._lstm_padding)
        self.new_input_conv = nn.Conv2d(
            self._lstm_input_channels, self._lstm_hidden_channels,
            self._lstm_kernel_size, padding=self._lstm_padding)
        self.forget_gate_conv = nn.Conv2d(
            self._lstm_input_channels, self._lstm_hidden_channels,
            self._lstm_kernel_size, padding=self._lstm_padding)
        self.output_gate_conv = nn.Conv2d(
            self._lstm_input_channels, self._lstm_hidden_channels,
            self._lstm_kernel_size, padding=self._lstm_padding)

        self.u_deconv = nn.ConvTranspose2d(self._lstm_hidden_channels,
                                           self._u_channels, 4, stride=4)

        # For sampling the query image
        self.observation_factor_conv = nn.Conv2d(
            self._u_channels, image_channels,
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


class Grid(object):
    def __init__(self, channels, shape, bounds):
        self.tensor = torch.zeros(channels, *shape)
        self.shape = shape
        self.bounds = bounds
        self.length = (bounds[:, 1] - bounds[:,0])
        self.voxel_size = self.length / shape
        self.ndim = bounds.shape[0]

    def _voxel_index(self, point):
        idx = [-1, -1, -1]
        for i in range(self.ndim):
            if point[i] < self.bounds[i, 0] or point[i] >= self.bounds[i, 1]:
                return [-1, -1, -1]
            idx[i] = (point[i] - self.bounds[i, 0]) // self.voxel_size[i]
        return idx

    def _inside(self, point):
        return self._voxel_index(point) != [-1, -1, -1]

    def _setup_ray(self, ray_origin, ray_direction):
        NDIM = self.bounds.shape[0]

        def compute_ray_state(origin):
            voxel_idx = self._voxel_index(origin)
            voxel_steps = np.sign(ray_direction)
            offset_in_voxel = (origin - self.bounds[:, 0]) - (voxel_idx * self.voxel_size)
            t_max = np.zeros(NDIM)
            for i in range(NDIM):
                if voxel_steps[i] > 0:
                    t_max[i] = (self.voxel_size[i] - offset_in_voxel[i]) / ray_direction[i]
                elif voxel_steps[i] < 0:
                    t_max[i] = (offset_in_voxel[i] - self.voxel_size[i]) / ray_direction[i]
            t_delta = np.abs(ray_direction[0:3] / self.voxel_size)
            return (voxel_idx, voxel_steps, t_max, t_delta)

        max_t = [0 for _ in range(NDIM)]
        # Check if ray inside the box
        inside = True
        quadrant = [None for i in range(NDIM)]
        candidate_plane = [None for i in range(NDIM)]
        LEFT = -1
        MIDDLE = 0
        RIGHT = 1
        for i in range(NDIM):
            if ray_origin[i] < self.bounds[i, 0]:
                quadrant[i] = LEFT
                candidate_plane[i] = self.bounds[i, 0]
                inside = False
            elif ray_origin[i] > self.bounds[i, 1]:
                quadrant[i] = RIGHT
                candidate_plane[i] = self.bounds[i, 1] - 1e-5
                inside = False
            else:
                quadrant[i] = MIDDLE
        # Inside box, so just return voxel coordinate
        if inside:
            return ray_origin, compute_ray_state(ray_origin)
        # Calculate T distances to candidate planes
        for i in range(NDIM):
            if quadrant[i] != MIDDLE and ray_direction[i] != 0:
                max_t[i] = (candidate_plane[i] - ray_origin[i]) / ray_direction[i]
            else:
                max_t[i] = -1

        # Get largest of the max_t's for final choice of intersection
        which_plane = 0
        for i in range(1, NDIM):
            if max_t[which_plane] < max_t[i]:
                which_plane = i

        # Check final candidate actually inside box
        if max_t[which_plane] < 0:
            # Didn't hit, so just return same origin
            return ray_origin, None

        new_origin = np.array([0, 0, 0])
        for i in range(NDIM):
            if which_plane != i:
                new_origin[i] = (
                    ray_origin[i] + max_t[which_plane] * ray_direction[i])
                # Check if inside box
                if (new_origin[i] < self.bounds[i, 0] or
                    new_origin[i] > self.bounds[i, 1]):
                    return ray_origin, None
            else:
                new_origin[i] = candidate_plane[i]
        return new_origin, compute_ray_state(ray_origin)

    def _next_ray(self, ray_origin, ray_direction, ray_state):
        voxel_idx, step, t_max, t_delta = ray_state
        NDIM = self.bounds.shape[0]
        t_min = float('inf')
        for d in range(NDIM):
            if step[d] == 0:
                continue
            offset = (ray_origin[d] - self.bounds[d, 0]) % self.voxel_size[d]
            if step[d] > 0:
                diff = self.voxel_size[d] - offset
            else:
                diff = offset
            t = diff / np.abs(ray_direction[d])
            if t < t_min:
                t_min = t
        new_origin = ray_origin + ray_direction[0:3] * (t_min + 1e-10)
        return new_origin
        # if t_max[0] < t_max[1]:
        #     if t_max[0] < t_max[2]:
        #         x = x + step[0]
        #         if x == just_out_x:
        #             pass
        #         t_max[0] += t_delta[0]
        #     else:
        #         z = z + step[2]
        #         if z == just_out_z:
        #             pass
        #         t_max[2] += t_delta[2]
        # else:
        #     if t_max[1] < t_max[2]:
        #         y = y + step[1]
        #         if y == just_out_y:
        #             pass
        #         t_max[1] += t_delta[1]
        #     else:
        #         z = z + step[2]
        #         if z == just_out_z:
        #             pass
        #         t_max[2] += t_delta[2]

    def trace(self, ray_origin, ray_direction):
        ray_pos, ray_state = self._setup_ray(ray_origin, ray_direction)
        if ray_state is None:
            return []
        voxel_idxs = []
        while True:
            voxel_idx = self._voxel_index(ray_pos)
            if voxel_idx == [-1, -1, -1]:
                break
            # Add voxel to voxel_idxs
            voxel_idxs.append(voxel_idx)
            # Trace to next voxel
            ray_pos = self._next_ray(ray_pos, ray_direction, ray_state)
        return voxel_idxs


class Renderer(nn.Module):
    def __init__(self, image_shape, input_shape, pose_channels=7, model_settings={}, train=True):
        super(Renderer, self).__init__()

        self._query_image_channels = 3

        self._image_shape = image_shape
        self._input_shape = input_shape
        self._pose_channels = pose_channels
        self._representation_channels = input_shape[1]
        self._lstm_layers = 6
        self._train = train

        self.fn_generator = DRAWFunctionGenerator(input_shape, model_settings, train=train)
        self.generator = Generator(input_shape, pose_channels)

    def observation_distribution(self, u, sigma):
        return self.generator.observation_distribution(u, sigma)

    def z_distribution_with_loss(self, representation, target_representation=None):
        return self.fn_generator(representation, target_representation)

    def z_distribution(self, representation, target_representation=None):
        return self.fn_generator(representation, target_representation)[0]

    def sample_z(self, representation, target_representation=None):
        return self.z_distribution(representation, target_representation)[0].rsample()

    def encode(self, context_image, context_pose):
        # Produce a feature and depth map
        pass

    def init_grid(self, shape, bounds):
        # Encoded image channels + 1 for stopping probability
        channels = self._input_shape[1] + 1
        grids = Grid(channels, shape, bounds)
        return grids

    def trace(self, grids, features, depth, pose, ax):
        # For each point in feature/depth maps, trace a ray
        # through the grid
        features_cpu = features.cpu()
        batch_size = features.shape[0]
        height = features.shape[3]
        width = features.shape[4]
        fov = 1.57
        for b in range(batch_size):
            if b == 0:
                pass
                #print(pose[b, :, 0, 0])
            grid = grids[b]

            q = q_mult(axisangle_to_q((0, 1, 0), -np.pi / 2), pose[b, 3:, 0, 0])
            R = quat2mat(q)
            #R = np.linalg.inv(R)
            t = pose[b, 0:3, 0, 0]
            t = t
            near = 0.1
            far = 10000
            inv_tan_ang = 1.0 / np.tan(fov / 2)
            persp_t = np.array([
                [inv_tan_ang, 0.0, 0.0, 0.0],
                [0.0, inv_tan_ang, 0.0, 0.0],
                [0.0, 0.0, far / (far - near), -far * near / (far - near)],
                [0.0, 0.0, 1.0, 0.0]
            ])
            camera_to_screen = persp_t
            screen_to_camera = np.linalg.inv(camera_to_screen)
            dx = (screen_to_camera * np.array([1.0, 0.0, 0.0, 1.0]) -
                  screen_to_camera * np.array([0.0, 0.0, 0.0, 1.0]))
            dy = (screen_to_camera * np.array([0.0, 1.0, 0.0, 1.0]) -
                  screen_to_camera * np.array([0.0, 0.0, 0.0, 1.0]))
            RC = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])

            #print('tracing {:d} rays...'.format(height * width))
            start = time.time()
            avg_intersections = 0
            for y in range(0, height):
                for x in range(0, width):
                    # Screen space -> raster space
                    u = (x + 0.5) / width * 2.0 - 1.0
                    v = (y + 0.5) / height * 2.0 - 1.0
                    # Raster space point -> camera space point
                    ray_dir = np.dot(screen_to_camera, np.array([u, v, 0, 1]))
                    # Turn into a vector
                    ray_dir[3] = 0.0
                    ray_dir /= np.linalg.norm(ray_dir)
                    ray_dir = -np.dot(R, ray_dir)
                    ray_origin = t.cpu().numpy()

                    # Trace ray through grid
                    voxel_idxs = grid.trace(ray_origin, ray_dir)
                    avg_intersections += len(voxel_idxs)
                    if len(voxel_idxs) == 0:
                        continue
                    idxs = [[], [], []]
                    for idx in voxel_idxs:
                        idxs[0].append(int(idx[0]))
                        idxs[1].append(int(idx[1]))
                        idxs[2].append(int(idx[2]))
                    # Insert features at grid positions
                    grid.tensor[:256, idxs[0], idxs[1], idxs[2]] += (
                        features_cpu[b, 0, :, y, x][:, np.newaxis])

                    # Draw single ray
                    #if b == 0 and x == width // 2 and y == height // 2:
                    # Draw frustum of rays
                    #if b == 0 and x % 4 == 0 and y % 4 == 0:
                    #    ax.quiver(ray_origin[0], ray_origin[1], ray_origin[2],
                    #              ray_dir[0], ray_dir[1], ray_dir[2], length=0.3)
            #print('time {:.3f}, avg intersections {:f}'.format(
            #    time.time() - start,
            #    avg_intersections / (height * width)))


    def project(self, grid, pose):
        # Produce a feature and depth map
        pass

    def forward(self, global_z, query_pose, query_image=None, cuda=False,
                debug=False):
        '''
        Perform inference of the query_image using the generator network.
        '''
        batch_size = query_pose.shape[0]
        hidden_g, state_g, u_g = self.generator.init(batch_size)

        if cuda:
            hidden_g = hidden_g.cuda()
            state_g = state_g.cuda()
            u_g = u_g.cuda()

        # Perform the projection operation on the global_z

        for layer in range(self._lstm_layers):
            # Update generator lstm state
            hidden_g, state_g, u_g = self.generator(query_pose, global_z, hidden_g, state_g, u_g)

        return u_g
