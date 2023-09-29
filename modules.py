"""A library of modules for building the Conformer."""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import torch
import torch.nn as nn


# Activation functions


class GLU(nn.Module):
	
	def __init__(self, dim):
		super(GLU, self).__init__()
		self.dim = dim
	
	def forward(self, x):
		return nn.functional.glu(x, dim=self.dim)


# Tensor manipulation

class Transpose(nn.Module):
	
	def __init__(self, dim1, dim2):
		super(Transpose, self).__init__()
		self.dim1 = dim1
		self.dim2 = dim2
	
	def forward(self, x):
		return x.transpose(self.dim1, self.dim2)


# Layers


class Linear(nn.Module):

	def __init__(self, input_dim, output_dim, device='cpu'):
		super(Linear, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim).to(device)
		nn.init.xavier_uniform_(self.linear.weight)
		nn.init.zeros_(self.linear.bias)

	def forward(self, x):
		return self.linear(x)


class MLPLayer(nn.Module):

	def __init__(self, input_dim, output_dim, activation='ReLU', device='cpu'):
		super(MLPLayer, self).__init__()
		self.linear = Linear(input_dim, output_dim, device)
		if activation == 'ReLU': self.activate = nn.ReLU()
		elif activation == 'Sigmoid': self.activate = nn.Sigmoid()
		elif activation == 'GLU': self.activate = GLU()

	def forward(self, x):
		return self.activate(self.linear(x))
		

