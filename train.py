from settings import CONFIG
from settings import CONFIG
from models import MODEL

config = CONFIG("./settings/config.json")
model = MODEL(config)
model.train()