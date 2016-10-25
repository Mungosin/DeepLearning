from tensorflow.examples.tutorials.mnist import input_data
from tf_deep import TFDeep
from confusion_matrix import get_confusion_matrix
from confusion_matrix import eval_AP
from confusion_matrix import eval_perf_multi

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def predict_c1_class_tf(X):
    def classify(X):
        return tfLog.eval(X)[:,1]
    return classify
#with tf.device("/gpu:0"):
C=2
layers = [784,10]

tfDeep = TFDeep(layers,param_delta=0.01,param_lambda=0.01,std=0.001,checkpoint_file="model-(784,10).ckpt")
tfDeep.train(mnist.train.images,mnist.train.labels,10000,print_every=100)
tfDeep.save()

probs = tfDeep.eval(mnist.train.images)
# recover the predicted classes Y
my_predictions = probs[:,1] >0.5
my_predictions = my_predictions.flatten()


# graph the decision surface
bbox=(np.min(input_data, axis=0), np.max(input_data, axis=0))

#axis limits of the graph
C=10
possible_labels = np.array(range(C))
mat = get_confusion_matrix(np.argmax(probs,axis=1),np.argmax(mnist.train.labels,axis=1),possible_labels,True,format_length=10)
accuracy, prec, rec = eval_perf_multi(mat)
# AP_c0 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,0].argsort()],0)
# AP_c1 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,1].argsort()],1)
# AP_c2 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,2].argsort()],2)
# AP_c3 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,3].argsort()],3)
# AP_c4 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,4].argsort()],4)
# AP_c5 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,5].argsort()],5)
# AP_c6 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,6].argsort()],6)
# AP_c7 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,7].argsort()],7)
# AP_c8 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,8].argsort()],8)
# AP_c9 = eval_AP(np.argmax(mnist.train.labels,axis=1)[probs[:,9].argsort()],9)
print
print "Accuracy"
print accuracy
print
print "Precision"
print prec
print
print "Recall"
print rec
print
print "Average Precision"
print "[0]-%.2f, [1]-%.2f, [2]-%.2f, [3]-%.2f, [4]-%.2f, [5]-%.2f, [6]-%.2f, [7]-%.2f, [8]-%.2f, [9]-%.2f" %(AP_c0,AP_c1,AP_c2,AP_c3,AP_c4,AP_c5,AP_c6,AP_c7,AP_c8,AP_c9)