"""A trainable vowel detector using MLP"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import sys, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn
import features, preprocessor, modules, timit_vowels

class model(nn.Module):

	def __init__(self, preprocessor_dims, output_dim, mode, device='cpu'):
		super(model, self).__init__()
		self.mode = mode
		self.preprocessor = preprocessor.MLP(preprocessor_dims, device)
		self.linear = modules.Linear(preprocessor_dims[-1], output_dim, device)
		if mode == 'binary':
			self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		y = self.preprocessor(x)
		y = self.linear(y)
		if self.mode == 'binary':
			y = self.sigmoid(y)
			y = y.reshape(y.shape[0], -1)
		return y

def train(net, optimizer, x, y, mode):
	if mode == 'binary':
		lf = nn.BCELoss()
	else:
		lf = nn.CrossEntropyLoss()
	optimizer.zero_grad()
	y_ = net(x)
	if mode == 'binary':
		loss = lf(y_, y)
	else:
		# cross-entropy loss compares (B, C, N) with (B, N)
		loss = lf(y_.transpose(1, 2), y)
	loss.backward()
	optimizer.step()
	return loss


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='BIO') # binary or BIO
	parser.add_argument('--train')
	parser.add_argument('--save')
	parser.add_argument('--load')
	parser.add_argument('--test')
	parser.add_argument('--num_epochs', default=0, type=int)
	parser.add_argument('--seq_len', default=100, type=int)
	parser.add_argument('--batch_size', default=100, type=int)
	args = parser.parse_args()

	device = torch.device('cpu')
	if torch.cuda.is_available: device = torch.device('cuda')

	if args.mode == 'binary':
		output_dim = 1
	elif args.mode == 'BIO':
		output_dim = 3

	if args.load is None:
		m = model([25, 1000, 1000], output_dim, args.mode, device=device)
	else:
		m = torch.load(args.load)
		
	if not args.train is None:
		start_time = time.time()
		optimizer = torch.optim.Adam(m.parameters())
		trn_wavs = glob.glob(args.train + '*.WAV')
		trn_phns = glob.glob(args.train + '*.PHN')
		trn_wavs.sort()
		trn_phns.sort()
		indices = np.arange(len(trn_wavs))
		rng = np.random.default_rng()
		for e in range(args.num_epochs):
			loss = 0
			num_data = 0
			rng.shuffle(indices)
			ip = 0
			while ip < len(trn_wavs):
				x, y, ip = timit_vowels.get_batch(trn_wavs, trn_phns, indices, ip, args.batch_size, args.seq_len, args.mode)
				x, y = x.to(device=device), y.to(device=device)
				loss += train(m, optimizer, x, y, args.mode)
				num_data += y.shape[0] * y.shape[1]
			if args.num_epochs < 10:
				print(e + 1, loss / num_data)
			elif (e + 1) % (args.num_epochs // 10) == 0:
				print(e + 1, loss / num_data)
		end_time = time.time()
		sys.stderr.write('# Training complete in ' + str(end_time - start_time) + ' seconds\n')
		if not args.save is None:
			torch.save(m, args.save)
			sys.stderr.write('# Model saved as ' + args.save + '\n')

	if not args.test is None:
		wavs = glob.glob(args.test + '*.WAV')
		phns = glob.glob(args.test + '*.PHN')
		wavs.sort()
		phns.sort()
		indices = np.arange(len(wavs))
		ip = 0
		num_data = 0
		num_correct = 0
		while ip < len(wavs):
			x, y, ip = timit_vowels.get_batch(wavs, phns, indices, ip, args.batch_size, args.seq_len, args.mode)
			x, y = x.to(device=device), y.to(device=device)
			y_ = m(x)
			if args.mode == 'binary':
				y_ = y_ > 0.5
			elif args.mode == 'BIO':
				y_ = y_.argmax(dim=-1)
			num_correct += (y_ == y).sum()
			num_data += y.shape[0] * y.shape[1]
		print(num_correct / num_data)

