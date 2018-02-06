import tensorflow as tf 

class Model(object): 
    def __init__(self, classifier):
        self.classifier = classifier 

    def cross_entropy_loss(self, y):
        with tf.name_scope('loss'): 
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.classifier.get_scores())
            self.loss = tf.reduce_mean(entropy, name='loss')
    
    def optimizer(self, learning_rate, global_step): 
        with tf.name_scope('optimize'): 
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
