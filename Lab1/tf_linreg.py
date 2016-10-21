import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_data = [1,2,3,4,5]
output_data = [3.2,4.8,7.2,8.8,11.2]
## 1. definicija računskog grafa
# podatci i parametri
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

polovina = tf.constant(0.5,tf.float32)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak  * 0.5 da se derivacijom rijesi ova polovina i bude better
loss = polovina* (Y-Y_)**2 * (1./len(input_data))

# optimizacijski postupak: gradijentni spust learning rate ubije sveee aaaaa
trainer = tf.train.GradientDescentOptimizer(0.01)
train_op = trainer.minimize(loss)

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())
y_valid = Y
#gradient definišn
grads_and_vars = trainer.compute_gradients(loss, var_list=[a,b],)
apply_grads = trainer.apply_gradients(grads_and_vars)
## 3. učenje
# neka igre počnu!
for i in range(100):
    
    #value = sess.run([grads_and_vars], feed_dict={X: [1], Y_: [3]})
    #print value
    #print "vrijednost izlaza", sess.run([Y], feed_dict={X: [1]})
    
    #moj analiticki gradijent :D
    #print (sess.run([Y], feed_dict={X: [1]})[0] - 3)* 1
    
    sess.run(apply_grads,feed_dict={X: input_data, Y_: output_data})
    #print i
#     test tf.Printa
#     test = tf.constant([1.0,2.0])
#     test2 = tf.Print(test,[test],message=("test varijablica "))
#     sess.run([test2]) #printa u terminal :'( suzice
    
#     print i
#     print
#     val_loss, _, val_a,val_b = sess.run([loss, train_op, a,b], 
#         feed_dict={X: [1,2,3,4], Y_: [3,5,7,9]})
    #print(i,val_loss, val_a,val_b)
print
print sess.run([Y], feed_dict={X:input_data})
print sess.run([y_valid], feed_dict={X:input_data})
plt.scatter(input_data,output_data)
plt.plot(input_data,sess.run([Y], feed_dict={X:input_data})[0])
plt.show()
sess.close()