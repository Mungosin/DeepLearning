import tensorflow as tf
class TFDeep:
    def __init__(self, layer, param_delta=0.01, param_lambda=1e-4):
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
           - param_delta: training step
        """
        
        self.sess = tf.Session()
        
        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        # ...
        self.X = tf.placeholder(tf.float32, [None,layer[0]])
        self.Yoh_ = tf.placeholder(tf.float32, [None,layer[-1]])
        prev = layer[0]
        out = self.X
        
        #Makni kad prebacis u pycharm
        l2_reg= None
        for dim in layer[1:-1]:
            temp_W = tf.Variable(tf.random_normal([prev,dim]))
            temp_b = tf.Variable(tf.random_normal([dim]))
            out = tf.nn.relu(tf.matmul(out,temp_W)+temp_b)
            prev = dim
            if l2_reg == None:
                l2_reg = tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
            else:
                l2_reg += tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
            
            
        temp_W = tf.Variable(tf.random_normal([prev,layer[-1]]))
        temp_b = tf.Variable(tf.random_normal([layer[-1]]))
        out = tf.nn.softmax(tf.matmul(out,temp_W)+temp_b)
        if l2_reg != None:
            l2_reg += tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
        else:
            l2_reg = tf.nn.l2_loss(temp_W) + tf.nn.l2_loss(temp_b)
            
        self.Y = out

        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        # ...
        #Y - izlaz
        #Yoh - onehot 
        self.loss = (tf.reduce_mean(-tf.reduce_sum(tf.log(self.Y)*self.Yoh_,reduction_indices=1)
                        + param_lambda*l2_reg))

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
        return self.sess.run(self.Y,feed_dict={self.X:X})
        pass
    
    def __del__(self):
        self.sess.close()