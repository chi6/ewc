import tensorflow as tf
from trainer_l2 import Trainer
from data_handler import DataHandler
from config import get_config

#get config parameters
config, unparsed = get_config()

# Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.
print('Loading data...')
data_handler = DataHandler(config.dataset)
mnist, mnist_2 = data_handler.split_dataset() # Train on 1-4
# mnist_2, mnist = data_handler.split_dataset() # Train on 5-9

# Retrain/test the network on MNIST 5-9
trainer = Trainer(retrain=True,config=config)

trainer.train(mnist_2)
trainer.test(mnist_2)