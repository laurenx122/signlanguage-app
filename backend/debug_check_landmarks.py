import numpy as np

seq = np.load("data/processed_sequences/1/12.npy")

print("Shape:", seq.shape)
print("Min:", seq.min())
print("Max:", seq.max())
print("Mean:", seq.mean())
