from tensorflow.keras.utils import plot_model
from utils.utils import get_callbacks
from data.loader import LOADER

class BASE():
	def __init__(self, config, model_name):
		self.config = config
		self.model_name = model_name
		self.config.CONFIG_INFO['MODEL_IMAGE_PATH'] += self.model_name+'.png'
		self.model = self.compose_model()

	def train(self):
		self.init_loaders()
		self.model.fit_generator(
			self.train_loader,
			validation_data=self.valid_loader,
			epochs=self.config.HYPERPARAMETERS['NUM_EPOCHS'],
			callbacks=get_callbacks(self.config, self.model_name)
		)

	def save(self, path):
		self.model.save(path)

	def load_weights(self, path=''):
		pth = path if path != '' else self.config.BEST_WEIGHT_PATH
		self.model.load_weights(pth)

	def init_loaders(self):
		self.train_loader = LOADER(self.config, 'train')
		self.valid_loader = LOADER(self.config, 'valid')

	def summary(self):
		self.model.summary()

	def plot(self):
		plot_model(self.model, self.config.MODEL_IMAGE_PATH, show_shapes=True)

	def predict(self, data):
		return self.model.predict(data)[0]

	def predict_on_batch(self):
		batch = self.TEST_LOADER(self.config.TEST_BATCH_SIZE, 'test')
		return self.model.predict_on_batch(batch)

	def compose_model(self):
		return

	def process(self):
		return