import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y

def data_augument(x, y, shuffle):
    x1 = tf.convert_to_tensor(x,dtype=tf.float32)
    y1 = tf.convert_to_tensor(y,dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([x1,y1],
                  shuffle=shuffle, capacity=10*100)
    img = input_queue[0]
    lab = input_queue[1]


    img = tf.reshape(img,[32, 32, 3])
    img = tf.cast(img, tf.float32)
    # pre_ processing
    if shuffle:
        img = tf.image.random_flip_left_right(img)

    img = tf.image.per_image_standardization(img)


    img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1,
                                          batch_size=100, capacity=10 * 100)


    return img_batch, lab_batch


def weight_variable(shape,std):
    initial = tf.truncated_normal(shape,mean=0,stddev=std,dtype=tf.float32)
    return tf.Variable(initial)


def batch_norm(inputs, is_training,is_conv_out=False,decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)


def CNN_model(sample , label,is_train ):
    features = sample

    y = label
    x = tf.reshape(features,[-1,32,32,3])
    #Conv_1, output -1,16,16,32
    W_conv1 = weight_variable([5,5,3,64],0.01)
    bias_con1 = tf.constant(0.1,shape=[64])
    hc1 = tf.nn.conv2d(x,W_conv1,strides=[1,1,1,1],padding='SAME')+bias_con1
    h_conv1_batch = batch_norm(hc1, is_train,is_conv_out=True)
    h_conv_1 = tf.nn.relu(h_conv1_batch)
    h_conv_1 = tf.nn.max_pool(h_conv_1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

    #Conv_2  output -1,8,8,128
    W_conv2 = weight_variable([5,5,64,128],0.01)
    bias_conv2= tf.constant(0.1,shape=[128])
    hc2 = tf.nn.conv2d(h_conv_1,W_conv2,strides=[1,1,1,1],padding='SAME')+bias_conv2
    h_conv2_batch = batch_norm(hc2, is_train,is_conv_out=True)
    h_conv_2 = tf.nn.relu(h_conv2_batch)
    h_conv_2 = tf.nn.max_pool(h_conv_2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    #Conv_3  output -1,8,8,128
    W_conv3 = weight_variable([3,3,128,64],0.05)
    bias_con3 = tf.constant(0.1,shape=[64])
    hc3 = tf.nn.conv2d(h_conv_2,W_conv3,strides=[1,1,1,1],padding='SAME')+bias_con3
    h_conv3_batch = batch_norm(hc3, is_train,is_conv_out=True)
    h_conv_3 = tf.nn.relu(h_conv3_batch)
    h_conv_3 = tf.nn.max_pool(h_conv_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #reshape to output -1, 4*4*64
    h_conv_4 = tf.reshape(h_conv_3,[-1,4*4*64])

    #softmax layer
    W_fc2 = weight_variable([4*4*64,10],0.1)
    bias_fc2 = tf.constant(0.2,shape=[10])
    output = tf.matmul(h_conv_4,W_fc2)+bias_fc2

    #output1 =tf.reshape(tf.arg_max(tf.nn.sigmoid(output),1),[-1,1])
    #print (tf.shape(output1))
    #loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
    gradient_layer1 = tf.reduce_max(tf.abs((tf.gradients(loss, W_conv1))))
    gradient_final = tf.reduce_max(tf.abs((tf.gradients(loss,W_conv3))))
    train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(output, 1)), tf.float32))

    return  loss, gradient_layer1, gradient_final, train, accuracy

def train(is_train,iteration,x,y,testx,testy):


    with tf.Session() as sess:

        gradient_layer1_plot = []
        gradient_final_plot = []
        iteration_plot = []
        training = []
        if is_train:
            loss, gradient_layer1, gradient_final, train, accuracy = CNN_model(x , y,is_train )
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            max_acc = 0
            for i in range(iteration):


                accuracyout,gradient_layer1_out,gradient_final_out,_  = sess.run([accuracy,gradient_layer1,gradient_final,train])


                # print(predout)
                training.append(accuracyout)
                gradient_layer1_plot.append(gradient_layer1_out)
                gradient_final_plot.append(gradient_final_out)
                iteration_plot.append(i)
                if (i+1)%100 ==0:
                    print ('iteration ', i + 1, ' accuracy ', accuracyout)
                if accuracyout > max_acc:
                    max_acc = accuracyout
                    saver.save(sess, 'hw2part2/cifar10.ckpt', global_step=i + 1)

        else:
            loss, gradient_layer1, gradient_final, train, accuracy = CNN_model(testx, testy,is_train )
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            model_file = tf.train.latest_checkpoint('hw2part2/')
            saver.restore(sess, model_file)

            test_accuracy = 0
            n = 100
            for iter in range(n):

                result = sess.run(accuracy)
                test_accuracy += result
            test_accuracy /= n

            print("Accuracy: {:.4f}".format(test_accuracy))

        coord.request_stop()
        coord.join(threads)
        sess.close()
        return iteration_plot, gradient_final_plot, gradient_layer1_plot,training


if __name__ == '__main__':

    x, y = load_CIFAR_batch('devkit/data_batch_1')

    testx, testy = load_CIFAR_batch('devkit/test_batch')

    #data augument
    x,y = data_augument(x,y,True)
    testx,testy = data_augument(testx,testy, False)

    iteration_plot, gradient_final_plot, gradient_layer1_plot ,training= train(True, 2000,x,y,testx,testy)
    #iteration_plot, gradient_final_plot, gradient_layer1_plot = train(False, 2000,x,y,testx,testy)
    plt.plot(iteration_plot,training)
    plt.ylabel('training accuracy')
    plt.xlabel('iterations')
    plt.show()
'''
    fig = plt.figure()
    ax1 = plt.subplot(2,2,1)
    ax1.set_title('layer1 gradient magnitude', fontsize=18)
    ax2 = plt.subplot(2,2,2)
    ax2.set_title('final gradient magnitude', fontsize=18)
    ax1.plot(iteration_plot, gradient_layer1_plot)
    ax2.plot(iteration_plot, gradient_final_plot)
'''
