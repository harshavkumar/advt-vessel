import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile

def create_dsets():
	loader_file = "C:/Users/Tarsh/Downloads/Compressed/leftImg8bit/sample.csv"
	load_from = "C:/Users/Tarsh/Downloads/Compressed/leftImg8bit/train_extra/"
	fl = pd.read_csv(loader_file)

	all_files = fl['x'].tolist()

	all_indices = np.arange(len(all_files))

	train_ratio = 0.9

	np.random.shuffle(all_indices)

	train_size = int(train_ratio*len(all_indices))

	print("Training Size : {}".format(train_size))

	valid_size = len(all_indices) - train_size
	print("Validation Size : {}".format(valid_size))

	train_idxs = []
	valid_idxs = []

	print('Creating Training Dataset...')
	while len(train_idxs) < train_size:
		idx = np.random.randint(0, len(all_indices))
		if idx not in train_idxs:
			train_idxs.append(idx)
			file = all_files[all_indices[idx]]
			path = load_from+file
			copyfile(path, 'C:/Users/Tarsh/Downloads/Compressed/leftImg8bit/data/train/{}'.format(file[file.rindex('/')+1:]))

	print('Creating Valid Dataset')
	while len(valid_idxs) < valid_size:
		idx = np.random.randint(0, len(all_indices))
		if idx not in train_idxs and idx not in valid_idxs:
			valid_idxs.append(idx)
			file = all_files[all_indices[idx]]
			path = load_from+file
			copyfile(path, 'C:/Users/Tarsh/Downloads/Compressed/leftImg8bit/data/valid/{}'.format(file[file.rindex('/')+1:]))

if __name__ == "__main__":
	create_dsets()