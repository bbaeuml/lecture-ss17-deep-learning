
from load_mnist import load_mnist
import numpy as np


def softmax(a):
    """computes the softmax function of the tensor of shape (N_examples, K_classes)
    resulting in a  tensor of shape [N_examples, K_classes] of the normalized probabilites for each class.
    """
    e = np.exp(a)
    return np.transpose(np.transpose(e)/e.sum(axis=1))


def features(x):
    """computes the (simple linear) feature mapping of a tensor of shape 
    [N_examples, dim_x] by adding a constant feature resulting in 
    a tensor of shape [N_examples, dim_x + 1].
    """
    def helper_fun(z):
        return np.insert(z, 0, 1.)

    return np.apply_along_axis(helper_fun, 1, x)

def activity(W, phi):
    """computes the activity
    Args:
      W: weight matrix of shape (dim_phi, K_classes)
      phi: feature tensor of shape (N_examples, dim_phi)
    Returns:
      Activity tensor of shape (N_examples, K_classes)
    """
    return softmax(np.dot(phi, W))

def cost(y, t):
    """computes the cost from the predicted probabilities y target variables t. 
 Both tensors are of shape (N_examples, K_classes).
    """
    return np.mean(np.sum(-(t * np.log(y)), 1))

def dcost(y, t, phi):
    """computes the gradient of the cost wrt. the weights
    Args:
      y, t: the predicted probability and target variable tensors of shape (N_examples, K_classes)
      phi: feature tensor of shape (N_examples, dim_phi)

    Returns:
      The gradient tensor of shape (dim_phi, K_classes).
    """
    return np.tensordot(phi, (y - t), axes=([0],[0]))/phi.shape[0]

def one_hot_coding(c, K):
    """computes the one-out-of-K coding for a tensor of shape (N_examples) resulting
 in a tensor of shape (N_examples, K).
    """    
    return np.eye(K)[c]

def accuracy(W, x, t):
    """computes the accuracy of the model with  the weights W (shape (dim_phi, N_classes)) 
for a test set  x (shape (N_examples, dim_x)) and t (shape (N_examples, K_classes)).
    """
    phi = features(x)
    y = activity(W, phi)

    correct  = np.equal(np.argmax(y, axis=1), np.argmax(t, axis=1))
    return np.mean(correct)
    

def main():

    train_data, validate_data, test_data  = load_mnist('mnist.pkl.gz')

    # design matrix of shape (num_examples, dim_x); dim_x = 784
    x_all = train_data[0]
    num_examples = x_all.shape[0]

    # label matrix  (N x 1)
    c_all = train_data[1]
    
    # feature mapping phi(x) resulting in shape (num_examples, dim_phi); dim_phi = dim_x + 1
    phi_all = features(x_all)
    dim_phi = phi_all.shape[1]
    
    K = 10 # number of classes
    # target variable  (num_examples, K)
    t_all = one_hot_coding(c_all, K)
    
    # the same for the test data
    test_x = test_data[0]
    test_c = test_data[1]
    test_t = one_hot_coding(test_c, K)
    

    batch_size = 600
    # learning rate
    eta = 0.13
    max_epochs = 100


    # weight matrix (dim_phi, K); initialized with 0
    W = np.zeros((dim_phi, K))

    # report initial accuracy
    print("initial,  test_accuracy: ", accuracy(W, test_x, test_t))
    
    # training loop
    for epoch in xrange(max_epochs):
        # in each new epoch randomly shuffle the training data
        perm = np.random.permutation(num_examples)
        phi_all = phi_all[perm]
        t_all = t_all[perm]

        # run through the mini batches and update gradient for each
        for end_index in xrange(batch_size, num_examples, batch_size):
            start_index = end_index - batch_size
            phi_batch = phi_all[start_index:end_index]
            t_batch = t_all[start_index:end_index]
            
            # activity or predicted probability (batch_size, K)
            y = activity(W, phi_batch)

            # gradient of the cost (dim_phi, K)
            dc = dcost(y, t_batch, phi_batch)

            # wheight update
            W = W - eta * dc
            
        print("epoch: {0},  test_accuracy: {1}".format(epoch, accuracy(W, test_x, test_t)))

main()
