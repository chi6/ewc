import tensorflow as tf 
from trainer_ewc import Trainer 
from data_handler import DataHandler

# Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.
print('Loading data...')
data_handler = DataHandler('mnist')
mnist, mnist_2 = data_handler.split_dataset() # Train on 1-4
# mnist_2, mnist = data_handler.split_dataset() # Train on 5-9

# Retrain/test the network on MNIST 5-9
trainer = Trainer(retrain=True)
trainer.restore() 
trainer.model.compute_fisher(dataset=mnist.validation.images, sess=trainer.sess, num_samples=200)
trainer.model.save_weights()
trainer.set_ewc_loss() 
trainer.define_summary() 
trainer.train(mnist_2) 
trainer.test(mnist_2) 