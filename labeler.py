"""Label vowels in file."""

import sys, glob, argparse, re
import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio
from vowel_detector_mlp import model
import features

def load(wavfile, device='cpu'):
	"""Load wavfile into a tensor."""
	x, fs = features.load(wavfile)
	x = features.extract(x, fs)
	x = x.unsqueeze(0)
	return x.to(device)

def rewrite_label(bio):
	"""Rewrite BIO label."""
	out = 'other'
	if bio == 1 or bio == 2:
		out = 'vowel'
	return out

def bias_model_output(y_):
	"""Bias model output."""
	y_ = torch.nn.functional.softmax(y_, dim=-1)
	bias = torch.tensor([0.2, -0.1, -0.1]).to(y_.device)
	bias = torch.tile(bias, (y_.shape[0], y_.shape[1], 1))
	y_ += bias
	return y_

def timit_output(y_, frame_shift=0.01, frame_size=0.025):
	"""Rewrite model output in TIMIT PHN style:
	- Labels in y_ mean: 0 for "O", 1 for "B", 2 for "I"
	"""
	y_ = bias_model_output(y_)
	y_ = y_.argmax(dim=-1).flatten()
	t = [0, 0, 0]
	out = []
	for i in range(len(y_)):
		now = round(i * frame_shift, 3)
		if t[-1] == y_[i] or (t[-1] == 1 and y_[i] == 2):
			t[1] = now
		else:
			t[-1] = rewrite_label(t[-1])
			out.append(t)
			t = [t[1], now, y_[i]]
	t[1] = (len(y_) - 1) * frame_shift
	t[-1] = rewrite_label(t[-1])
	out.append(t) 
	return out

def filter_output_by_length(output, threshold=0.1):
	"""Filter TIMIT-style output by length:
	- Only keep vowels whose length > threshold in seconds.
	"""
	for i in range(len(output)):
		s, e, l = output[i]
		if (e - s) < threshold and l == 'vowel':
			output[i][-1] = 'other'
	

def load_phn(phn, fs):
	"""Load TIMIT PHN file as reference."""
	out = []
	with open(phn) as f:
		for line in f:
			s, e, l = line.strip().split()
			s = float(s) / fs
			e = float(e) / fs
			if re.match('[aeiouAEIOU]', l):
				l = 'vowel'
			else:
				l = 'other'
			out.append([s, e, l])
	return out

def plot(x, fs, labels, fig, i):
	"""Plot annotated waveform."""
	ax = fig.add_subplot(2, 1, i)
	t = np.arange(len(x)) / fs
	ax.plot(t, x)
	for s, e, l in labels:
		if l == 'vowel':
			ax.axvspan(s, e, color='orange', alpha=0.1)
	return fig

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model')
	parser.add_argument('--wavs')
	parser.add_argument('--phns', action='store_true')
	parser.add_argument('--plot', action='store_true')
	parser.add_argument('--extract_to')
	args = parser.parse_args()
	m = torch.load(args.model)
	device = torch.device('cpu')
	if torch.cuda.is_available(): device = torch.device('cuda')
	for wav in glob.glob(args.wavs):
		x = load(wav, device)
		hyp = timit_output(m(x))
		filter_output_by_length(hyp)
		for s, e, l in hyp: print(s, e, l)
		if not args.extract_to is None:
			vi = 1
			x, fs = features.load(wav)
			for s, e, l in hyp:
				if l == 'vowel':
					duration = e - s
					st = s + 0.25 * duration
					et = e - 0.25 * duration
					xi = x[:, int(st * fs) : int(et * fs)]
					vfn = re.split(r'[\\/]', wav)[-1]
					vfn = re.sub('\.(wav|WAV)', '', vfn)
					vfn += '_V' + str(0) * (3 - len(str(vi))) + str(vi)
					vfn += '.wav'
					vfn = args.extract_to + '/' + vfn
					torchaudio.save(vfn, xi, fs, True)
					vi += 1
		if args.plot:
			x, fs = features.load(wav)
			fig = plt.figure()
			plot(x[-1], fs, hyp, fig, 1)
			if args.phns:
				phn = re.sub('WAV$', 'PHN', wav)
				ref = load_phn(phn, fs)
				plot(x[-1], fs, ref, fig, 2)
			plt.show()
