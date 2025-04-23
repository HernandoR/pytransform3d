"""Constants of rotation module."""

# import numpy as np
import torch


unitx = torch.tensor([1.0, 0.0, 0.0])
unity = torch.tensor([0.0, 1.0, 0.0])
unitz = torch.tensor([0.0, 0.0, 1.0])

R_id = torch.eye(3)
a_id = torch.tensor([1.0, 0.0, 0.0, 0.0])
q_id = torch.tensor([1.0, 0.0, 0.0, 0.0])
q_i = torch.tensor([0.0, 1.0, 0.0, 0.0])
q_j = torch.tensor([0.0, 0.0, 1.0, 0.0])
q_k = torch.tensor([0.0, 0.0, 0.0, 1.0])
p0 = torch.tensor([0.0, 0.0, 0.0])

eps = 1e-7

two_pi = 2.0 * torch.pi
half_pi = 0.5 * torch.pi
