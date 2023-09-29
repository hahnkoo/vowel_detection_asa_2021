"""Preprocessor Library"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import torch
import torch.nn as nn
import modules

class MLP(nn.Module):

	"""An MLP preprocessor"""
	
	def __init__(self, dims, device='cpu'):
		super(MLP, self).__init__()
		self.layers = [modules.MLPLayer(dims[i], dims[i + 1], device=device) for i in range(len(dims) - 1)]

	def forward(self, x):
		y = x
		for layer in self.layers:
			y = layer(y)
		return y


class CNN(nn.Module):

	"""A CNN preprocessor"""

	def __init__(self, input_dim, kernel_size, num_filters, device='cpu'):
		super(CNN, self).__init__()
		p = (kernel_size - 1) // 2
		self.filter = nn.Sequential(
			nn.LayerNorm(input_dim).to(device),
			nn.Conv2d(1, num_filters, kernel_size, padding=p).to(device),
			nn.ReLU() )
	
	def forward(self, x):
		x = x.unsqueeze(1)
		y = self.filter(x)
		y = y.transpose(1, 2)
		y = y.reshape(y.shape[0], y.shape[1], -1)
		return y


if __name__ == '__main__':
	x = torch.rand((5, 100, 25))
	m = CNN(3, 10)
	y_ = m(x)
	print(y_.shape)	
