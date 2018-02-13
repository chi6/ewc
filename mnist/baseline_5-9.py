from trainer import Trainer 
from data_handler import DataHandler

# Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.
print('Loading data...')
data_handler = DataHandler('mnist')
# mnist, mnist_2 = data_handler.split_dataset() # Train on 1-4
mnist_2, mnist = data_handler.split_dataset() # Train on 5-9

# Train the network on only MNIST 5-9 
trainer = Trainer(retrain=False)
trainer.train(mnist) 
trainer.test(mnist) 