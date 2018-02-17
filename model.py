import tensorflow as tf 
import numpy as np 

class Model(object): 
    def __init__(self, classifier, config):
        self.classifier = classifier
        self.config = config


    def compute_fisher(self, x, dropout, dataset, sess, num_samples=200): 
        """
        Compute Fisher information matrix
        """

        # Initialize Fisher information matrix 
        self.F_matrix = [] 
        self.variable_list = tf.trainable_variables() 
        for var in range(len(self.variable_list)): 
            self.F_matrix.append(np.zeros(self.variable_list[var].get_shape().as_list()))
        
        # Sample from a random class from softmax 
        scores = tf.nn.softmax(self.classifier.get_scores())
        class_ind = tf.to_int32(tf.multinomial(tf.log(scores), 1)[0][0])
        print('Class index: %s', class_ind)
        # x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_placeholder')

        with sess.as_default(): 
            sess.run(tf.global_variables_initializer())
            test = sess.run(tf.log(scores[0, class_ind]), feed_dict={dropout: 0.75}) 
            print(test)

            # Compute Fisher information matrix 
            for idx in range(num_samples): 
                # Select input image randomly 
                image_idx = np.random.randint(dataset.shape[0])
                print(image_idx)

                if hasattr(self, "variable_list"): 
                    print("Has variables")
                    print(len(self.variable_list))
                else: 
                    print("No variables")

                # Compute first-order derivatives
                # Consider using log likelihood as an alternative implementation  

                # derivatives = sess.run(tf.gradients(tf.log(scores[0, class_ind]), self.variable_list), feed_dict={x: dataset[image_idx:image_idx + 1]})

                # log_likelihood = self.cross_entropy_loss(self.classifier.get_scores())

                # Square the derivatives and add to the total 
                for var in range(len(self.F_matrix)):
                    self.F_matrix[var] += np.square(derivatives[var])

        # Divide by the total number of sample 
        for var in range(len(self.F_matrix)): 
            self.F_matrix[var] /= num_samples 

    
    def save_weights(self, sess):
        """
        Save weights after training source task. 
        """
        if not hasattr(self, "variable_list"): 
            self.variable_list = tf.trainable_variables() 
        
        self.star_vars = []

        with sess.as_default(): 
            for v in range(len(self.variable_list)):
                self.star_vars.append(self.variable_list[v].eval())
    
    def ewc_loss(self, y, star_vars, fisher_multiplier): 
        """
        Elastic weight consolidation. 
        """
        self.ewc_loss = self.cross_entropy_loss(y)

        for var in range(len(self.variable_list)): 
            self.ewc_loss += (fisher_multiplier / 2) * tf.reduce_sum(tf.multiply(self.F_matrix[var].astype(np.float32),tf.square(self.variable_list[var] - self.star_vars[var])))  

    def l2_loss(self,y):
        """
        L2 loss. adjust lambda in config
        """
        self.l2_loss = self.cross_entropy_loss(y)

        for var in range(len(self.variable_list)):
            self.l2_loss += self.config.l2_lambda*tf.nn.l2_loss(self.star_vars[var]-self.variable_list[var])

    def cross_entropy_loss(self, y):
        with tf.name_scope('loss'): 
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.classifier.get_scores())
            self.loss = tf.reduce_mean(entropy, name='loss')
        return self.loss

    def optimizer(self, learning_rate, global_step, train_vars):
        # NOTE: when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op.(https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #select optimizer
        if self.config.optimizer == 'sgd':
            optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif self.config.optimizer == 'adam':
            optim = tf.train.AdamOptimizer(learning_rate)
        else:
            raise Exception("[!] Caution! Don't use {} opimizer.".format(self.config.optimizer))

        with tf.control_dependencies(update_ops), tf.name_scope('optimize'): 
            self.optimizer = optim.minimize(self.loss, global_step=global_step, var_list=train_vars)

    def optimizer_ewc(self, learning_rate, global_step, train_vars): 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # select optimizer
        if self.config.optimizer == 'sgd':
            optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif self.config.optimizer == 'adam':
            optim = tf.train.AdamOptimizer(learning_rate)
        else:
            raise Exception("[!] Caution! Don't use {} opimizer.".format(self.optimizer))

        with tf.control_dependencies(update_ops), tf.name_scope('optimize'):
            self.optimizer = optim.minimize(self.ewc_loss, global_step=global_step, var_list=train_vars)

