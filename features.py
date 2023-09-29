"""Library of functions for feature extraction"""

__author__ = "Hahn Koo (hahn.koo@sjsu.edu)"

import sys
import torch, torchaudio

def extract(x, sampling_rate):
	"""Extract features."""
	num_frames = len(x[-1]) // int(0.01 * sampling_rate)
	fbank = torchaudio.compliance.kaldi.fbank(x, sample_frequency=sampling_rate, snip_edges=False)[:num_frames]
	pitch = torchaudio.functional.compute_kaldi_pitch(x, sample_rate=sampling_rate, snip_edges=False)[-1, :num_frames, :]
	x = torch.cat((fbank, pitch), dim=-1)
	return x 

def load(fn):
	return torchaudio.load(fn)


if __name__ == '__main__':
	x, fs = load(sys.argv[1])
	x = extract(x, fs)
	print(x.shape)
