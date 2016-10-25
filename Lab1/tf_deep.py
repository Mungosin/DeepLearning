import tensorflow as tf
from sklearn.cross_validation import train_test_split
import math
import random

class TFDeep:
    def __init__(self, layer, param_delta=0.01, param_lambda=1e-4, activation=tf.nn.relu,
                 std=1.0, optimizer=tf.train.GradientDescentOptimizer, batch_norm = False,
                 batch_decay = 0.999, checkpoint_file="model.ckpt"):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """
        self.reset()
        self.checkpoint_file = checkpoint_file
        self.sess = tf.Session()
        self.weights = []
        self.biases = []

        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        # ...
        self.X = tf.placeholder(tf.float32, [None, layer[0]], name="input")
        self.Yoh_ = tf.placeholder(tf.float32, [None, layer[-1]], name="output")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        prev = layer[0]
        out = self.X

        # Makni kad prebacis u pycharm
        l2_reg = None
        for dim in layer[1:-1]:
            temp_W = tf.Variable(tf.random_normal([prev, dim], stddev=std))
            temp_b = tf.Variable(tf.random_normal([dim], stddev=std))
            out = tf.matmul(out, temp_W) + temp_b
            if batch_norm:
                out = self.batch_norm_wrapper(out,self.phase_train, decay=batch_decay)
            out = activation(out)
            prev = dim
            if l2_reg == None:
                l2_reg = tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
            else:
                l2_reg += tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
            self.weights.append(temp_W)
            self.biases.append(temp_b)

        temp_W = tf.Variable(tf.random_normal([prev, layer[-1]], stddev=std), name="last_layer_w")
        temp_b = tf.Variable(tf.random_normal([layer[-1]], stddev=std), name="last_layer_b")
        self.weights.append(temp_W)
        self.biases.append(temp_b)

        self.last_layer_w = temp_W

        out = tf.nn.softmax(tf.matmul(out, temp_W) + temp_b)
        if l2_reg != None:
            l2_reg += tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
        else:
            l2_reg = tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)

        self.Y = out

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        # ...
        # Y - izlaz
        # Yoh - onehot
        self.loss = (tf.reduce_mean(-tf.reduce_sum(tf.log(self.Y) * self.Yoh_, reduction_indices=1)
                                    + param_lambda * l2_reg))

        # formulacija operacije ucenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        # ...
        self.trainer = optimizer(param_delta)
        self.optimizer = self.trainer.minimize(self.loss)
        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # ...

        pass

    def train(self, X, Yoh_, param_niter, print_every=1000, early_stop_after=50, minibatch=False):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
           - early_stop_after: if False it will not early stop if it is a number the number will define patience
           - minibatch: if False it will not train in minibatches, otherwise the number will define number of batches
        """
        # incijalizacija parametara
        #   koristiti: tf.initialize_all_variables
        # ...
        X_train, X_validate, y_train, y_validate = train_test_split(X, Yoh_, test_size=0.2, stratify=Yoh_)
        self.sess.run(tf.initialize_all_variables())
        val_best = self.sess.run([self.loss], feed_dict={self.X: X_validate, self.Yoh_: y_validate, self.phase_train:False})
        cnt = 0
        num_layers = len(self.weights)
        temp_weights = [0] * num_layers
        temp_biases = [0] * num_layers

        for i in range(param_niter):

            if minibatch != False:
                self.train_mb(X_train, y_train, minibatch)
            else:
                _ = self.sess.run([self.optimizer], feed_dict={self.X: X_train, self.Yoh_: y_train, self.phase_train:True})

            loss = self.sess.run([self.loss], feed_dict={self.X: X_train, self.Yoh_: y_train, self.phase_train:False})
            if i % print_every == 0:
                print "Iteration = %d, Loss = %s" % (i, loss)
            if early_stop_after != False:
                val_loss = self.sess.run([self.loss], feed_dict={self.X: X_validate, self.Yoh_: y_validate, self.phase_train:False})
                if val_best < val_loss:
                    if cnt >= early_stop_after:
                        print "Early stopping on iteration = %d, training model restored from checkpoint with training loss = %s, validation loss = %s" % (
                        i, loss, val_loss)
                        # self.restore()
                        for i in range(num_layers):
                            self.weights[i].assign(temp_weights[i])
                            self.biases[i].assign(temp_biases[i])
                        print "Restored model validation set loss = %s" % (
                        self.sess.run([self.loss], feed_dict={self.X: X_validate, self.Yoh_: y_validate, self.phase_train:False}))
                        return
                    cnt += 1
                else:
                    cnt = 0
                    # self.save()
                    for i in range(num_layers):
                        temp_weights[i] = self.sess.run(self.weights[i])
                        temp_biases[i] = self.sess.run(self.biases[i])
                    val_best = val_loss

    def train_mb(self, X, Yoh_, num_batches):
        batch_size = int(math.ceil(len(X) / num_batches))
        data = zip(X, Yoh_)
        random.shuffle(data)
        for X_minibatch, y_minibatch in self.next_batch(data, batch_size):
            _ = self.sess.run([self.optimizer], feed_dict={self.X: X_minibatch, self.Yoh_: y_minibatch, self.phase_train:True})

    def next_batch(self, data, batch_size):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(data), batch_size):
            yield zip(*data[i:i + batch_size])

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        return self.sess.run(self.Y, feed_dict={self.X: X, self.phase_train:False})
        pass

    def batch_norm_wrapper(self, inputs, is_training, decay = 0.999):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        
        
        
        def f_train():
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, 1e-3)
        def f_test():
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, 1e-3)
        
        normed = tf.cond(self.phase_train,f_train, f_test)
        return normed
    
    def get_last_layer_weights(self):
        return self.sess.run([self.last_layer_w])

    def save(self):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.checkpoint_file)

    def restore(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.checkpoint_file)

    def reset(self):
        tf.reset_default_graph()

    def __del__(self):
        self.sess.close()