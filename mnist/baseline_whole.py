from trainer import Trainer 
from data_handler import DataHandler
from config import get_config

#get config parameters
config, unparsed = get_config()

# Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.
print('Loading data...')
data_handler = DataHandler(config.dataset)
mnist = data_handler.get_dataset() # Train on entire dataset
print('Data is loaded.')

# Train the network on entire mnist data
trainer = Trainer(retrain=False,config=config)
trainer.train(mnist) 
trainer.test(mnist) 