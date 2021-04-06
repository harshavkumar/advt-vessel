import os
import pandas as pd
import keras

# class EVALCALL(keras.callbacks.Callback):
# 	def on_epoch_end(self):
# 		loader_file = "D:/morpheus/data/candidate-placements.csv"
# 		load_from = "D:/morpheus/data/test/"

# 		fl = pd.read_csv(loader_file)

# 		all_files = fl['#filename'].tolist()
# 		all_labels = fl['region_shape_attributes'].tolist()

# 		all_indices = []

# 		for i in range(len(all_files)):
# 			if all_files[i] in os.listdir(load_from):
# 				all_indices.append(i)


def get_callbacks(config, model='custom'):
	config.MODULE_BASE_PATH = 'C:/Users/Tarsh/Downloads/Compressed/leftImg8bit/'
	all_checks = os.listdir(config.MODULE_BASE_PATH+'checkpoints/')
	counter = 0
	max = -1

	for folder in all_checks:
			if 'checkpoints_{}'.format(model) in folder:
					if int(folder[folder.rindex('_')+1:]) > max:
							max = int(folder[folder.rindex('_')+1:])

	counter = max+1
	check_path = config.MODULE_BASE_PATH+'checkpoints/checkpoints_{}_{}/'.format(model, counter)
	logs_path = config.MODULE_BASE_PATH+'logs/logs_{}_{}/'.format(model, counter)

	if not os.path.isdir(check_path) and not os.path.isdir(logs_path):
			os.mkdir(check_path)
			os.mkdir(logs_path)


	checkpoint = keras.callbacks.ModelCheckpoint(
						check_path+'weights.{epoch:02d}-{loss:.2f}.hdf5',
						monitor='val_loss',
						verbose=0,
						save_best_only=True,
						save_weights_only=False
					)
	earlystop = keras.callbacks.EarlyStopping(
						monitor='val_loss',
						min_delta=0,
						patience=3,
						verbose=0
					)
	tensorboard = keras.callbacks.TensorBoard(
						log_dir='logs',
						histogram_freq=0,
						batch_size=32,
						write_graph=True,
						write_grads=True,
						write_images=True
					)
	reducelr = keras.callbacks.ReduceLROnPlateau(
						monitor='val_loss',
						factor=0.02,
						patience=1,
						verbose=0,
						mode='auto',
						min_delta=0.0001,
						cooldown=0,
						min_lr=0
					)

	return [checkpoint, tensorboard, reducelr, earlystop]