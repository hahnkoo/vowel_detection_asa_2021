"""Get vowel data from TIMIT."""

import sys, glob, re
import numpy as np
import torch, torchaudio
import features

def load_timit_wav(wav_file):
	"""Load TIMIT wav file."""
	x, fs = torchaudio.load(wav_file)
	return x, fs

def load_timit_phn(phn_file, sampling_rate):
	"""Load TIMIT PHN file."""
	vops = []; veps = []
	with open(phn_file) as f:
		for line in f:
			ll = line.strip().split()
			st, et = ll[:2]
			p = ''
			if len(ll) == 3: p = ll[2]
			st = float(st) / sampling_rate
			et = float(et) / sampling_rate
			if re.match('[aeiouAEIOU]', p):
				vops.append(st)
				veps.append(et)
	return np.array(vops), np.array(veps)

def label_frames(vops, veps, x, sampling_rate, stride=0.01, width=0.025, mode='binary'):
	"""Label each frame in either of two modes: (1) binary, (2) BIO
	(1) binary
	- 0 for "no", 1 for "yes"
	(2) BIO
	- 0 for "O", 1 for "B", 2 for "I"
	"""
	s = int(sampling_rate * stride)
	w = int(sampling_rate * width)
	n = len(x)
	num_frames = n // s
	y = np.zeros(num_frames)
	for i in range(len(vops)):
		sn = int(np.ceil(vops[i] / stride))
		en = int(np.ceil(veps[i] / stride))
		if mode == 'binary':
			y[sn:en] = 1
		elif mode == 'BIO':
			y[sn] = 1
			y[sn + 1 : en] = 2
	return y

def get_data(wav, phn, label_mode):
	"""Get data from wav and phn."""
	x, fs = features.load(wav)
	vops, veps = load_timit_phn(phn, fs)
	y = label_frames(vops, veps, x[-1], fs, mode=label_mode)
	if label_mode == 'binary':
		y = torch.tensor(y).type(torch.FloatTensor)
	elif label_mode == 'BIO':
		y = torch.LongTensor(y)
	x = features.extract(x, fs)
	return x, y

def split_sequence(x, y, seq_len):
	"""Split x and y each into a batch of segments of length seq_len."""
	short = seq_len - (len(y) % seq_len)
	x = torch.nn.functional.pad(x, (0, 0, 0, short))
	x = x.reshape(-1, seq_len, x.shape[-1])
	y = torch.nn.functional.pad(y, (0, short))
	y = y.reshape(-1, seq_len)
	return x, y 

def get_batch(wavs, phns, indices, index_position, batch_size, seq_len, label_mode):
	"""Get a batch of data from wavs and phns."""
	xs = torch.tensor([])
	if label_mode == 'binary':
		ys = torch.tensor([])
	elif label_mode == 'BIO':
		ys = torch.LongTensor([])
	batch_full = False
	while not batch_full:
		i = indices[index_position]
		x, y = get_data(wavs[i], phns[i], label_mode)
		x, y = split_sequence(x, y, seq_len)
		xs = torch.cat((xs, x))
		ys = torch.cat((ys, y))
		index_position += 1
		batch_full = ys.shape[0] > batch_size
		batch_full = batch_full or index_position >= len(wavs)
	return xs, ys, index_position

if __name__ == '__main__':
	wavs = glob.glob(sys.argv[1])
	phns = glob.glob(sys.argv[2])
	wavs.sort()
	phns.sort()
	indices = np.arange(len(wavs))
	rng = np.random.default_rng()
	rng.shuffle(indices)
	index_position = 10
	xs, ys, index_position = get_batch(wavs, phns, indices, index_position, 100, 100, 'BIO')
	print(xs.shape, ys.shape, index_position)
	print(ys.type())

