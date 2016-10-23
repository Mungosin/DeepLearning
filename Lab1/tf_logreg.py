import tensorflow as tf


class TFLogreg:
    def __init__(self, D, C, param_delta=0.01, param_lambda=1e-4, std=1.0):
        """Arguments:
           - D: dimensions of each datapoint
           - param_delta: training step
        """

        self.sess = tf.Session()

        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        # ...
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.W = tf.Variable(tf.random_normal([D, C], stddev=std))
        self.b1 = tf.Variable(tf.random_normal([C], stddev=std))

        # formulacija modela: izračunati self.probs
        #   koristiti: tf.matmul, tf.nn.softmax
        # ...
        self.model_score = tf.matmul(self.X, self.W) + self.b1
        self.model_softmax = tf.nn.softmax(self.model_score)  # vektor C
        self.Y = self.model_softmax

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        # ...
        # Y - izlaz
        # Yoh - onehot
        self.loss = (tf.reduce_mean(-tf.reduce_sum(tf.log(self.Y) * self.Yoh_, reduction_indices=1)
                                    + param_lambda * tf.nn.l2_loss(self.W)))

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        # ...
        self.trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.optimizer = self.trainer.minimize(self.loss)
        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # ...

        pass

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        #   koristiti: tf.initialize_all_variables
        # ...
        self.sess.run(tf.initialize_all_variables())
        for i in range(param_niter):
            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X, self.Yoh_: Yoh_})
        # optimizacijska petlja
        #   koristiti: tf.Session.run
        # ...
        pass

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        return self.sess.run(self.Y, feed_dict={self.X: X})
        pass

    def __del__(self):
        self.sess.close()