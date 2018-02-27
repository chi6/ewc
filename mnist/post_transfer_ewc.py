from trainer_ewc import Trainer 
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

# Test MNIST 0-4 post transfer
trainer = Trainer(retrain=True,config=config)
trainer.restore() 
trainer.model.save_weights(trainer.sess) 
trainer.model.compute_fisher(x=trainer.x, dataset=mnist.validation.images, sess=trainer.sess, num_samples=200)
trainer.set_ewc_loss() 
trainer.define_summary() 
trainer.restore() 
trainer.test(mnist) 