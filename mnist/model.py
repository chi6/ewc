import tensorflow as tf 
import numpy as np 

class Model(object): 
    def __init__(self, classifier, config):
        self.classifier = classifier
        self.config = config

    
    def compute_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]

    def compute_fisher(self, trainer, dataset, sess, num_samples=200): 
        """
        Compute Fisher information matrix
        """

        # Initialize Fisher information matrix 
        self.F_matrix = [] 
        self.variable_list = tf.trainable_variables() 
        for var in range(len(self.variable_list)): 
            self.F_matrix.append(np.zeros(self.variable_list[var].get_shape().as_list()))

        sess.run(tf.global_variables_initializer())


        # Sample from a random class from softmax 
        # scores = tf.nn.softmax_cross_entropy_with_logits(labels=trainer.y, logits=self.classifier.get_scores())
        scores = tf.nn.softmax(self.classifier.get_scores())
        class_ind = tf.to_int32(tf.multinomial(tf.log(scores), 1)
        [0][0]) 

        with sess.as_default(): 
            # Compute Fisher information matrix 
            for idx in range(num_samples): 
                # Select input image randomly 
                image_idx = np.random.randint(dataset.images.shape[0])

                results = sess.run(tf.log(scores), 
                feed_dict={
                    trainer.y: dataset.labels[image_idx:image_idx + 1],
                    trainer.x: dataset.images[image_idx:image_idx + 1], 
                    trainer.phase: 1, 
                    trainer.dropout: 0.75})
                # print(results)


                # Compute first-order derivatives
                # Consider using log likelihood as an alternative implementation 
                class_idx = np.random.randint(10)
                log_likelihood = tf.log(scores[0, class_ind])
                gradients = self.compute_gradients(-log_likelihood, self.variable_list) 

                derivatives = sess.run(gradients, 
                feed_dict={
                    trainer.y: dataset.labels[image_idx:image_idx + 1],
                    trainer.x: dataset.images[image_idx:image_idx + 1], 
                    trainer.phase: 1, 
                    trainer.dropout: 0.75})

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
        # Initialize loss function

        self.cross_entropy_loss(y)
        self.ewc_loss = self.loss
        
        for var in range(len(self.variable_list)): 
            self.ewc_loss += (fisher_multiplier / 2) * tf.reduce_sum(tf.multiply(self.F_matrix[var].astype(np.float32),tf.square(self.variable_list[var] - self.star_vars[var])))  

    def l2_loss(self,y, train_vars):
        """
        L2 loss. adjust lambda in config
        """
        self.cross_entropy_loss(y)
        self.l2_loss = self.loss

        for var in range(len(train_vars)):

            self.l2_loss += self.config.l2_lambda*tf.nn.l2_loss(train_vars[var])

    def cross_entropy_loss(self, y):
        with tf.name_scope('loss'): 
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.classifier.get_scores())
            self.loss = tf.reduce_mean(entropy, name='loss')
    
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
