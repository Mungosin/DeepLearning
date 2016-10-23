import numpy as np


def fcann2_train(X, Y_, std=1.0, learning_rate=0.5):
    input_size = 2
    hidden_size = 2
    output_size = 2
    n_iter = 10000

    W1 = std * np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = std * np.random.randn(hidden_size, output_size)
    b2 = np.zeros(output_size)

    Y_onehot = np.zeros((len(Y_), 2))
    Y_onehot[range(len(Y_)), Y_] = 1

    for i in range(n_iter):
        layer_1, layer_2, probs = fcann2_forward(X, W1, b1, W2, b2)
        if i % 2000 == 0:
            val = -np.log(probs) * Y_onehot
            loss = np.sum(val) / len(Y_)
            print "Iteration = %d Loss = %f" % (i, loss)
        # grad softmax
        dscores = probs
        dscores[range(len(Y_)), Y_] -= 1
        dscores /= len(Y_)

        # grad last layer
        grad_W2 = np.dot(layer_1.T, dscores)
        grad_b2 = np.sum(dscores, axis=0)

        # grad trough relu
        grad_trough_relu = np.dot(dscores, W2.T)
        grad_trough_relu[layer_1 <= 0] = 0

        # grad first layer
        grad_W1 = np.dot(X.T, grad_trough_relu)
        grad_b1 = np.sum(grad_trough_relu, axis=0)

        W2 += -learning_rate * grad_W2
        b2 += -learning_rate * grad_b2
        W1 += -learning_rate * grad_W1
        b1 += -learning_rate * grad_b1

    return W2, b2, W1, b1


def fcann2_forward(X, W1, b1, W2, b2):
    # Compute the forward pass
    scores = None
    l1 = X.dot(W1) + b1
    l1 = np.dot(X, W1) + b1
    l1[l1 <= 0] = 0  # ReLu
    scores = l1.dot(W2) + b2
    scores_exp = np.exp(scores)
    sum_exp = np.sum(np.exp(scores), axis=1)
    probabilities = scores_exp / sum_exp[:, None]
    return l1, scores, probabilities


def predict_c1_class(X, W1, b1, W2, b2):
    def classify(X):
        return fcann2_forward(X, W1, b1, W2, b2)[2][:, 1]

    return classify

