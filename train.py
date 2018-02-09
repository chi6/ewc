from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 
import tensorflow as tf
import utils
from tensorflow.examples.tutorials.mnist import input_data
from network import SimpleCNN
from model import Model 
from data_handler import DataHandler 

# Step 1: Read in data.
# Use TF Learn's built in function to load MNIST data to the folder 'data/mnist/'.  
print('Loading data...')
# mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)
data_handler = DataHandler('mnist')
# mnist = data_handler.get_dataset() 
mnist, mnist_2 = data_handler.split_dataset() 
retrain = False
print('Completed loading data.')

#####################################################################
#####################################################################

# Step 2: Define parameters for the model. 
N_CLASSES = 10 
LEARNING_RATE = 0.001 
BATCH_SIZE = 32
SKIP_STEP = 10 
DROPOUT = 0.75
N_EPOCHS = 1 

#####################################################################
#####################################################################

# Step 3: Create placeholders for features and labels.
# Each image is represented as a 1x784 tensor (28*28 pixels = 784 pixels). Define a placeholder for dropout probability. Use 'None' for shape so we can dynamically change the 'batch_size' while building the graph. 
with tf.name_scope('data'): 
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_placeholder')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_placeholder')

dropout = tf.placeholder(dtype=tf.float32, name='dropout')
print('Completed defining parameters.')

#####################################################################
#####################################################################

# Steps 4/5: Create weights and do inference. 
# MODEL: conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax 
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

print('Constructing Simple CNN graph...')
cnn = SimpleCNN(input=x, dropout=dropout, num_classes=N_CLASSES, skip_layer=[''], weights_path='DEFAULT')

#####################################################################
#####################################################################

# Step 6: Define the loss function. 
# Use softmax cross entropy with logits as the loss function. Compute mean cross entropy. Note that softmax is applied internally. 
print('Defining the model')
model = Model(cnn)

model.cross_entropy_loss(y)

# If retraining the data, select trainable variables 
var_list = tf.trainable_variables()

if retrain: 
	train_vars = [var for var in var_list if 'fc1'   in var.name]
	model.optimizer(learning_rate=LEARNING_RATE, global_step=global_step, train_vars=train_vars)
	mnist = mnist_2 
else: 
	model.optimizer(learning_rate=LEARNING_RATE, global_step=global_step, train_vars=var_list)
print('Completed defining the model.')

# Define summary of the model 
with tf.name_scope('summaries'): 
    tf.summary.scalar('loss', model.loss)
    tf.summary.histogram('histogram loss', model.loss)
    summary_op = tf.summary.merge_all()

# Make directories for checkpoints 
utils.make_dir('checkpoints')
utils.make_dir('checkpoints/convnet_mnist')

#####################################################################
#####################################################################

# Step 7/8: Train/test the model. 
with tf.Session() as sess:
    # Initialize variables  
    print('Beginning session...')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver() 

    # Visualize results using Tensorboard 
    writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)
    checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))

    # Restore previous checkpoint, if checkpoint already exists 
    # if checkpoint and checkpoint.model_checkpoint_path: 
    #     saver.restore(sess, checkpoint.model_checkpoint_path)

    # Load previously trained data, if retraining. 
    if retrain: 
    	saver.restore(sess, checkpoint.model_checkpoint_path)
    	print('Retraining the network.')
    else: 
    	print('Training the whole network.')
    
    initial_step = global_step.eval()

    start_time = time.time() 
    
    num_batches = int(mnist.train.num_examples / BATCH_SIZE)
    total_loss = 0.0

    # Train the model N_EPOCHS times 
    for index in range(initial_step, num_batches * N_EPOCHS): 
        x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
        
        _, loss_batch, summary = sess.run(
                            [model.optimizer, model.loss, summary_op], 
                            feed_dict={x: x_batch, y: y_batch, dropout: DROPOUT})
        
        writer.add_summary(summary, global_step=index)
        total_loss += loss_batch 

        # Print out average loss after 10 steps 
        if (index + 1) % SKIP_STEP == 0: 
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss/SKIP_STEP))
            total_loss = 0.0 
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

    print('Optimization Finished!')
    print('Total time: {0} seconds'.format(time.time()- start_time))

    # Test the model 
    num_batches = int(mnist.test.num_examples/BATCH_SIZE)
    total_correct_preds = 0

    for i in range(num_batches): 
        x_batch, y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run(
                            [model.optimizer, model.loss, model.classifier.get_scores()], 
                            feed_dict={x: x_batch, y: y_batch, dropout: 1.0})
        
        # Calculate the total correct predictions 
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))



