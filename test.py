import torch
import gqn_generator, gqn_encoder

tower = gqn_encoder.Tower()

image = torch.rand(1, 3, 64, 64)
view = torch.rand(1, 7, 1, 1)

representation = tower(image, view)

gen = gqn_generator.Generator(representation.shape)

hidden, state, u = gen.init()
z = torch.zeros(1, 64, 16, 16)

print('Initial state')
print(hidden.shape, state.shape, u.shape)

hidden, state, u = gen(view, representation, z, hidden, state, u)
print('Updated state')
print(hidden.shape, state.shape, u.shape)
