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

# Define parameters for the model.
N_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1

class Trainer(object):
    def __init__(self, retrain, config):
        self.config = config
        # Define parameters for the model.
        self.N_CLASSES = self.config.n_classes
        self.LEARNING_RATE = self.config.learning_rate
        self.BATCH_SIZE = self.config.batch_size
        self.SKIP_STEP = self.config.skip_step
        self.DROPOUT = self.config.dropout
        self.N_EPOCHS = self.config.epoch_step

        self.retrain = retrain
        self.create_parameters()
        self.construct_graph() 
        self.construct_model() 
        # self.define_summary() 
        self.sess = tf.Session()

    def create_parameters(self):
        # Create placeholders for features and labels.
        # Each image is represented as a 1x784 tensor (28*28 pixels = 784 pixels). Define a placeholder for dropout probability. Use 'None' for shape so we can dynamically change the 'batch_size' while building the graph.
        with tf.name_scope('data'):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_placeholder')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_placeholder')
            self.phase = tf.placeholder(tf.bool, name='phase')

            self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            print('Completed defining parameters.')
    
    def construct_graph(self): 
        # Create weights and do inference.
        # MODEL: conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax

        print('Constructing Simple CNN graph...')
        self.cnn = SimpleCNN(input=self.x, 
                             phase=self.phase,
                             dropout=self.dropout,
                             num_classes=self.N_CLASSES,
                             skip_layer=[''],
                             weights_path='DEFAULT')
    
    def construct_model(self):
        # Define the loss function.
        # Use softmax cross entropy with logits as the loss function. Compute mean cross entropy. Note that softmax is applied internally.
        print('Defining the model')
        self.model = Model(self.cnn, config=self.config)

    def set_ewc_loss(self):  
        self.model.ewc_loss(self.y, self.model.star_vars, 0)

        # If retraining the data, select trainable variables
        var_list = tf.trainable_variables()

        if self.retrain:
            self.train_vars = [var for var in var_list if 'fc1'    in var.name]
            self.model.optimizer_ewc(learning_rate=self.LEARNING_RATE, global_step=self.global_step, train_vars=self.train_vars)
            # mnist = mnist_2
        else:
            self.model.optimizer_ewc(learning_rate=self.LEARNING_RATE, global_step=self.global_step, train_vars=var_list)
        print('Completed defining the model.')

    def define_summary(self):
        # Define summary of the model
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.model.ewc_loss)
            tf.summary.histogram('histogram loss', self.model.ewc_loss)
            self.summary_op = tf.summary.merge_all()

        # Make directories for checkpoints
        utils.make_dir('checkpoints')
        utils.make_dir('checkpoints/convnet_mnist')

    def restore(self): 
        saver = tf.train.Saver() 
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
        saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def train(self, source): 
        # Train/test the model.
        # with tf.Session() as self.sess:
        with self.sess.as_default():  
            # Initialize variables
            print('Beginning session...')
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # Visualize results using Tensorboard
            writer = tf.summary.FileWriter('./graphs/convnet', self.sess.graph)
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))

            # Restore previous checkpoint, if checkpoint already exists
            # if checkpoint and checkpoint.model_checkpoint_path:
            #     saver.restore(sess, checkpoint.model_checkpoint_path)

            # Load previously trained data, if retraining.
            if self.retrain:
                saver.restore(self.sess, checkpoint.model_checkpoint_path)
                self.sess.run(self.global_step.assign(0))
                print('Retraining the network.')
            else:
                print('Training the whole network.')

            initial_step = self.global_step.eval()
            start_time = time.time()

            mnist = source 
            num_batches = int(mnist.train.num_examples /self.BATCH_SIZE)
            total_loss = 0.0

            # Train the model N_EPOCHS times
            for index in range(initial_step, num_batches * self.N_EPOCHS):
                x_batch, y_batch = mnist.train.next_batch(self.BATCH_SIZE)

                _, loss_batch, summary = self.sess.run(
                                    [self.model.optimizer, self.model.ewc_loss, self.summary_op],
                                    feed_dict={self.x: x_batch,
                                                self.y: y_batch,
                                                self.phase: 1,
                                                self.dropout: self.DROPOUT})

                writer.add_summary(summary, global_step=index)
                total_loss += loss_batch

                # Print out average loss after 10 steps
                if (index + 1) % self.SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss/self.SKIP_STEP))
                    total_loss = 0.0
                    saver.save(self.sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

            print('Optimization Finished!')
            print('Total time: {0} seconds'.format(time.time()- start_time))

    def test(self, target): 
        # Test the model
        with self.sess.as_default(): 
            print('Testing the model...')
            mnist = target 
            num_batches = int(mnist.test.num_examples/self.BATCH_SIZE)
            total_correct_preds = 0

            for i in range(num_batches):
                x_batch, y_batch = mnist.test.next_batch(self.BATCH_SIZE)
                _, loss_batch, logits_batch = self.sess.run(
                                    [self.model.optimizer, self.model.ewc_loss, self.model.classifier.get_scores()],
                                    feed_dict={self.x: x_batch, 
                                            self.y: y_batch,
                                            self.phase: 0, 
                                            self.dropout: 1.0})

                # Calculate the total correct predictions
                preds = tf.nn.softmax(logits_batch)
                correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_batch, 1))
                accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
                total_correct_preds += self.sess.run(accuracy)

            print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
        self.sess.close() 