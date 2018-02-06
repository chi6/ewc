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
        
        MODEL: conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax 
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
        print('Completed constructing graph of SimpleCNN.')

    def create_graph(self):
        # First layer 
        conv1 = conv2d(self.X, 5, 5, 32, 1, 1, name='conv1', padding='SAME')
        pool1 = max_pool(conv1, 2, 2, 2, 2, padding='SAME', name='pool1')
        
        # Second layer 
        conv2 = conv2d(self.X, 5, 5, 64, 1, 1, name='conv2', padding='SAME')
        pool2 = max_pool(conv1, 2, 2, 2, 2, padding='SAME', name='pool2')

        # Fully-connected layer 
        output_shape = int(pool2.get_shape()) 
        height = int(output_shape[1])
        width = int(output_shape[2])
        num_filters = int(output_shape[-1])
        fc1 = fc(input=pool2, input_features=height*width*num_filters, 1024, name='fc1')

        # Softmax (prediction) layer
        sigma = softmax(input=fc1, num_input_units=1024, name='softmax_linear')
    
    def conv2d(input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'): 
        # Get number of input channels 
        input_channels = int(input.get_shape()[-1])

        # Create lambda function for the convolution 
        convolve = lambda i, k: tf.nn.conv2d(input=i, 
                                             filter=k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

        # Create tf variables for the weights and biases of the conv layer 
        with tf.variable_scope(name) as scope: 
            kernel = tf.get_variable(name='kernels', 
                                    shape=[filter_height, filter_width, input_channels, num_filters],
                                    initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable(name='biases', 
                                     shape=[num_filters], 
                                     initializer=tf.truncated_normal_initializer())

            conv = convolve(input, kernel)
            return tf.nn.relu(conv + biases, name=scope.name)

    def max_pool(input, filter_height, filter_width, stride_x, stride_y, padding='SAME', name):
        with tf.variable_scope(name) as scope: 
            pool = tf.nn.max_pool(value=input,
                                  ksize=[1, filter_height, filter_width, 1],
                                  strides=[1, stride_x, stride_y, 1], 
                                  padding='SAME')
            return pool 
    
    def fc(input, input_features, num_output_units, name): 
        with tf.variable_scope(name) as scope: 
            # Input feature dimensions: 7 x 7 x 64
            # Number of output units in hidden layer: 1024  
            w = tf.get_variable(name='weights', 
                                shape=[input_features, num_output_units], 
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name='biases', 
                                shape=[num_output_units], 
                                initializer=tf.constant_initializer(0.0))
            
            # Reshape input to 2-d. 
            input = tf.reshape(input, [-1, input_features])
            fc = tf.nn.relu(tf.matmul(input, w) + b, name='relu') 

            # Apply dropout.
            fc = tf.nn.dropout(fc, dropout, name='relu_dropout')
            return fc 

    def softmax(input, num_input_units, name): 
        with tf.variable_scope(name) as scope: 
            w = tf.get_variable(name='weights', 
                                shape=[num_input_units, N_CLASSES],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name='biases', 
                                shape=[N_CLASSES], 
                                initializer=tf.random_normal_initializer())
            logits = tf.matmul(input, w) + b
            return logits  

