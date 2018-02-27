import tensorflow as tf 
from trainer_ewc import Trainer 
from data_handler import DataHandler
import time
from config import get_config

#get config parameters
config, unparsed = get_config()

# Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.
print('Loading data...')
data_handler = DataHandler(config.dataset)
mnist, mnist_2 = data_handler.split_dataset()  # Train on 1-4
# mnist_2, mnist = data_handler.split_dataset() # Train on 5-9

# Retrain/test the network on MNIST 5-9
trainer = Trainer(retrain=True,config=config)
trainer.restore() 

print('Computing fisher information matrix.')
start = time.time() 
trainer.model.compute_fisher(trainer=trainer, dataset=mnist.validation, sess=trainer.sess, num_samples=200)
end = time.time() 
print('Time to compute fisher matrix: %s', (end-start))

trainer.model.save_weights(trainer.sess)
# trainer.set_ewc_loss() 
# trainer.define_summary() 
# trainer.train(mnist_2) 
# trainer.test(mnist_2) 