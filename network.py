import tensorflow as tf 
import numpy as np 

class SimpleCNN(object): 
    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'): 
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - num_classes: int, number of classes of the new dataset
        - skip_layer: list of strings, names of the layers you want to reinitialize
        - weights_path: path string, path to the pretrained weights,
                        (if mnist_weights.npy is not in the same folder)
        """    

        # Parse input arguments into class variables 
        self.X = x 
        self.NUM_CLASSES = num_classes 
        self.KEEP_PROB = keep_prob 
        self.SKIP_LAYER = skip_layer 

        if weights_path == 'DEFAULT': 
            self.WEIGHTS_PATH = './data/mnist_weights.npy'
        else: 
            self.WEIGHTS_PATH = weights_path 
        
        # Create graph of the model 
        self.create_graph()

    def create_graph(self): 
        with tf.variable_scope('conv1') as scope: 
            # Define kernel parameters. 
            filter_height = 5 
            filter_width = 5 
            input_channels = 1 
            num_filters = 32 
            
            # Reshape the images to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
            images = tf.reshape(x, shape=[-1, 28, 28, 1])
            kernel = tf.get_variable(name='kernel', 
                                    shape=[filter_height, filter_width, input_channels, num_filters], 
                                    initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable(name='biases', 
                                    shape=[num_filters], 
                                    initializer=tf.truncated_normal_initializer())
            conv = tf.nn.conv2d(input=images, 
                                filter=kernel,
                                strides=[1, 1, 1, 1], 
                                padding='SAME')

            # Output dimensions: BATCH_SIZE x 28 x 28 x 32 
            conv1 = tf.nn.relu(conv + biases, name=scope.name)

        with tf.variable_scope('pool1') as scope: 
            # Output dimensions: BATCH_SIZE x 14 x 14 x 32 
            pool1 = tf.nn.max_pool(value=conv1, 
                                ksize=[1, 2, 2, 1], 
                                strides=[1, 2, 2, 1], 
                                padding='SAME')

        with tf.variable_scope('conv2') as scope: 
            # Kernel is now of size 5 x 5 x 32 x 64 
            filter_height = 5 
            filter_width = 5 
            input_channels = 32 
            num_filters = 64 

            kernel = tf.get_variable(name='kernels', 
                                    shape=[filter_height, filter_width, input_channels, num_filters],
                                    initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable(name='biases', 
                                    shape=[num_filters], 
                                    initializer=tf.truncated_normal_initializer()
                                    )
            conv = tf.nn.conv2d(input=pool1, 
                                filter=kernel, 
                                strides=[1, 1, 1, 1], 
                                padding='SAME')
            # Output dimensions: BATCH_SIZE x 14 x 14 x 64 
            conv2 = tf.nn.relu(conv + biases, name=scope.name)
            
        with tf.variable_scope('pool2') as scope: 
            # Output dimensions: BATCH_SIZE x 7 x 7 x 64
            pool2 = tf.nn.max_pool(value=conv2, 
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], 
                                padding='SAME')

        with tf.variable_scope('fc') as scope: 
            # Input feature dimensions: 7 x 7 x 64
            # Number of output units in hidden layer: 1024   
            input_features = 7 * 7 * 64
            num_output_units = 1024   
            w = tf.get_variable(name='weights', 
                                shape=[input_features, num_output_units], 
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name='biases', 
                                shape=[num_output_units], 
                                initializer=tf.constant_initializer(0.0))
            
            # Reshape pool2 to 2-d. 
            pool2 = tf.reshape(pool2, [-1, input_features])
            fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu') 

            # Apply dropout.
            fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

        with tf.variable_scope('softmax_linear') as scope: 
            num_input_units = 1024 
            w = tf.get_variable(name='weights', 
                                shape=[num_input_units, N_CLASSES],initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name='biases', 
                                shape=[N_CLASSES], 
                                initializer=tf.random_normal_initializer())
            logits = tf.matmul(fc, w) + b

