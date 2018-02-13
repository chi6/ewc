from trainer import Trainer 
from data_handler import DataHandler

# Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.
print('Loading data...')
data_handler = DataHandler('mnist')
mnist = data_handler.get_dataset() # Train on entire dataset
print('Data is loaded.')

# Train the network on entire mnist data 
trainer = Trainer(retrain=False)
trainer.train(mnist) 
trainer.test(mnist) 