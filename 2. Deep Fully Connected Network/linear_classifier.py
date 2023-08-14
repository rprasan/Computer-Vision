"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
Project credit: Justin Johnson, EECS 498/598
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)              #initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]              #column 'c' in 'W' is the weights for the cth class
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])           #.product of an X[i] with 'W' yields X[i]'s classification scores for all 'C' classes
        correct_class_score = scores[y[i]]#indexing the scores using the true label y[i] yields the classification score obtained for the true class
        for j in range(num_classes):
            if j == y[i]:                 #error margin computed only if current class is not the true class
                continue
            margin = scores[j] - correct_class_score + 1#delta = 1
            if margin > 0:                #if score for the true class > the score for the current class, max(0,sj-syi+1) yields 0
                loss += margin
                #######################################################################
                # TODO:                                                               #
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW. (part 1) Rather than first computing the loss and then    #
                # computing the derivative, it is simple to compute the derivative    #
                # at the same time that the loss is being computed.                   #
                #######################################################################
                # Replace "pass" statement with your code
                #'DATA' part of ∂W
                dW[:,j]+=X[i,:]           #∇wc(Li)
                dW[:,y[i]]+=-X[i,:]       #∇wyi(Li)
                #######################################################################
                #                       END OF YOUR CODE                              #
                #######################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # Replace "pass" statement with your code
    dW=(dW/num_train)+(2*reg*W)           #∂W=(1/N)∇W(DATA) + ∇W(REG)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    #Compute the scores
    N=X.shape[0]
    scores=torch.mm(X,W)                   #each X[n,:] . each W[:,c]. Resultant order is (NxC)
    corrClassScores=scores[torch.arange(scores.shape[0]),y]#each sample's correct class score. Order is (N,)
    #Compute the margins
    mask=torch.ones_like(scores)           #calculates margins only if c≠yi
    mask[torch.arange(scores.shape[0]),y]=0#each sample's true class mask = 0
    margins=mask*(scores-corrClassScores.view(-1,1)+1)
    #Compute the loss
    loss+=(torch.sum(margins[margins>0]))/N#DATA part
    loss+=(reg*torch.sum(W*W))             #regularized loss (add REG part to DATA part)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Replace "pass" statement with your code
    #Compute the gradient ∂L/∂W
    mask[margins<=0]=0                     #set masks of all negative margins to 0s
    cwyi=torch.sum(mask,dim=1)             ##times -X[i,:] is added to ∂W[:y[i]]=sum of all non-zero elements in ith row of 'mask'
    mask[torch.arange(N),y]=-cwyi          #update each sample's true class mask, which is ucrrently 0, with that class' sum of masks of all other classes
    dW+=(torch.matmul(X.t(),mask))/N       #DATA part = (1/N)∇W(DATA)
    dW+=(2*reg*W)                          #Add REG part, which is ∇W(REG)
    #dW.index_add_(1,y,(-cwyi*X).t())       #use 'cwyi' to weight samples and subtract weighted difference from '∂W'
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in     #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    #                                                                       #
    # Hint: Use torch.randint to generate indices.                          #
    #########################################################################
    # Replace "pass" statement with your code
    indices=torch.randint(0,num_train,(batch_size,))#generate 'batch_size' #samples
    X_batch,y_batch=X[indices,:],y[indices]         #the sampled batch
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # TODO: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # Replace "pass" statement with your code
        #Gradient descent
        W-=(learning_rate*grad)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Replace "pass" statement with your code
    #Get the predictions
    _,y_pred=torch.max(torch.matmul(X,W),dim=1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO:   add your own hyper parameter lists.                             #
    ###########################################################################
    # Replace "pass" statement with your code
    #Choices of α and λ
    learning_rates=[1e-3,5e-3,1e-2,2e-2,3e-2,4e-2]
    regularization_strengths=[5e-3,1e-2,5e-2,1e-1]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first                     #
    ###########################################################################
    # Feel free to uncomment this, at the very beginning,
    # and don't forget to remove this line before submitting your final version
    # Replace "pass" statement with your code
    #Train the linear classifier
    _=cls.train(data_dict['X_train'], data_dict['y_train'],lr,reg,num_iters)
    #Classifier accuracy on the training set
    yHat=cls.predict(data_dict['X_train'])
    train_acc=(yHat==data_dict['y_train']).double().mean().item()*100
    #Classifier accuracy on the validation set
    yHat=cls.predict(data_dict['X_val'])
    val_acc=(yHat==data_dict['y_val']).double().mean().item()*100
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    N,C=X.shape[0],W.shape[1]
    #Loop over each sample in X
    for i in range(N):
        scores=W.t().mv(X[i,:])    #.product of an X[i] with 'W' yields X[i]'s classification scores for all 'C' classes
        corrClassScore=scores[y[i]]#indexing the scores using the true label y[i] yields the classification score obtained for the true class
        #The loss Li for sample X[i]
        lnC=-torch.max(scores)     #constant used for numerical stability of 'softmax'
        P=torch.exp(corrClassScore+lnC)/torch.sum(torch.exp(scores+lnC))#P(Y=yi|X=xi)
        loss+=-torch.log(P)        #Li=-ln(P(Y=yi|X=xi))
        #Loop over each class to compute the 'DATA' part of the gradient ∂L/∂W
        for j in range(C):
            #The 'DATA' part of the gradient ∂Li/∂W for sample X[i]
            dW[:,j]+=(torch.exp(scores[j]+lnC)/torch.sum(torch.exp(scores+lnC)))*X[i,:]#∇wc(Li)
            if y[i]==j:dW[:,y[i]]+=-X[i,:]
    #Compute the loss L
    loss/=N                        #'DATA' part of the loss L
    loss+=(reg*torch.sum(W*W))     #regularized loss (add 'REG' part to 'DATA' part)
    #Compute the gradient ∂L/∂W
    dW/=N                          #DATA part of ∂W
    dW+=(2*reg*W)                  #∂W=(1/N)∇W(DATA) + ∇W(REG)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    N=X.shape[0]
    #Compute the scores
    scores=torch.mm(X,W)                      #each X[n,:] . each W[:,c]. Resultant order is (NxC)
    corrClassScores=scores[torch.arange(N),y] #each sample's correct class score. Order is (N,)
    #Compute the probabilities P(Y=yi|X=xi); ∀i
    lnC,_=torch.max(scores,dim=1)             #constants used for numerical stability of 'softmax'
    lnC=torch.negative(lnC)
    P=torch.exp(corrClassScores+lnC)/torch.sum(torch.exp(scores+lnC.view(-1,1)),dim=1)
    #Compute the loss
    loss+=(torch.sum(-torch.log(P))/N)        #DATA part
    loss+=(reg*torch.sum(W*W))                #regularized loss (add REG part to DATA part)
    #Generate the mask
    expScores=torch.exp(scores+lnC.view(-1,1))#e^(sj+ln(C)) for the whole dataset. Rows and columns in 'expScores' represent samples and classes respectively
    mask=expScores/torch.sum(expScores,dim=1).view(-1,1)#corresponds to ∂Li/∂Wj=(e^(sji)/Σje^(sji))X[i,:]
    mask[torch.arange(N),y]+=-1               #corresponds to ∂Li/∂Wyi=-X[i,:] when j=y[i]
    #Compute the gradient ∂L/∂W
    dW+=(torch.matmul(X.t(),mask))/N          #DATA part = (1/N)∇W(DATA)
    dW+=(2*reg*W)                             #Add REG part, which is ∇W(REG)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    #Choices of α and λ
    learning_rates=[5e-3,1e-2,6e-2,7e-2,8e-2,9e-2]
    regularization_strengths=[1e-3,5e-3,1e-2,5e-2]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths