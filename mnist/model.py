import tensorflow as tf 

class Model(object): 
    def __init__(self, classifier):
        self.classifier = classifier 

    def cross_entropy_loss(self, y):
        with tf.name_scope('loss'): 
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.classifier.get_scores())
            self.loss = tf.reduce_mean(entropy, name='loss')
    
    def optimizer(self, learning_rate, global_step, train_vars):
        # NOTE: when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op.(https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops), tf.name_scope('optimize'): 
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step, var_list=train_vars)
