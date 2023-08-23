"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        import torch.nn.functional as F
        #Get the I/P and filter dimensions
        N,C,H,W=x.shape
        Fl,_,HH,WW=w.shape
        #Convolution parameters
        P,S=conv_param['pad'],conv_param['stride']
        #Zero-pad the I/P images along the H & W dimensions. Resultant
        #dimensions are (N,C,H+(2*conv_param['pad']),W+(2*conv_param['pad']))
        xCopy=x.clone()
        xCopy=F.pad(xCopy,(P,P,P,P))                    #default padding value is 0
        #Create the activation maps after calculating their H & W dimensions
        Hp=int(((H-HH+(2*P))/S)+1)                      #H'=((H-HH+2P)/S)+1; ignore the fractional part
        Wp=int(((W-WW+(2*P))/S)+1)                      #W'=((W-WW+2P)/S)+1; ignore the fractional part
        aMap=torch.empty((N,Fl,Hp,Wp),dtype=x.dtype,device=x.device)#activation map dimension is (#images,#filters,H',W')
        #Forward pass for convolution begins
        for n in range(N):                              #loop over each image
            for f in range(Fl):                         #loop over each filter
                i=j=0                                   #used to index the H & W dimensions of the activation maps
                for ht in range(0,xCopy.shape[2],S):    #loop over H'
                    if ht+HH>xCopy.shape[2]:break       #break if kernel's bottom edge extends beyond xCopy's H-limit
                    for wd in range(0,xCopy.shape[3],S):#loop over W'
                        if wd+WW>xCopy.shape[3]:break   #break if kernel's right edge extends beyond xCopy's W-limit
                        aMap[n,f,i,j]=torch.sum(xCopy[n,:,ht:ht+HH,wd:wd+WW]*w[f,:,:,:])+b[f]#convolution
                        j+=1                            #increment 'j' to advance along aMap's W'-dimension
                    i+=1                                #increment 'i' to advance along aMap's H'-dimension
                    j=0                                 #set 'j' to 0 after advancing to a new row along the H'-dimension
        out=aMap
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        import torch.nn.functional as F
        x,w,b,params=cache
        #Get the dimensions
        N,C,H,W=x.shape
        Fl,_,HH,WW=w.shape
        _,_,Hp,Wp=dout.shape
        #Convolution parameters
        P,S=params['pad'],params['stride']
        #Create the derivative variables
        xCopy=x.clone()
        xCopy=F.pad(xCopy,(P,P,P,P))                            #default padding value is 0
        dx=torch.zeros_like(xCopy,dtype=x.dtype,device=x.device)#same shape as 'xCopy'; don't use 'torch.empty()' as it intializes using the range [0,e-38]. So if initial value is greater than e-10, comparison check will fail
        dw=torch.zeros_like(w,dtype=w.dtype,device=w.device)    #same shape as 'w'; don't use 'torch.empty()' for the same reason
        #Backpropagation begins
        db=torch.tensor([torch.sum(dout[:,f,:,:]) for f in range(Fl)],dtype=b.dtype,device=b.device)#similar to FC networks, 'db' for a filter is the sum of all activation map elements corresponding to that filter and across all images
        for n in range(N):                                      #loop over each image
            for f in range(Fl):                                 #loop over each filter
                i=j=0                                           #used to index the H & W dimensions of the derivatives of the activation maps
                for ht in range(0,xCopy.shape[2],S):            #loop over H'
                    if ht+HH>xCopy.shape[2]:break               #break if kernel's bottom edge extends beyond xCopy's H-limit
                    for wd in range(0,xCopy.shape[3],S):        #loop over W'
                        if wd+WW>xCopy.shape[3]:break           #break if kernel's right edge extends beyond xCopy's W-limit
                        dw[f,:,:,:]+=xCopy[n,:,ht:ht+HH,wd:wd+WW]*dout[n,f,i,j]
                        dx[n,:,ht:ht+HH,wd:wd+WW]+=w[f,:,:,:]*dout[n,f,i,j]
                        j+=1                                    #increment 'j' to advance along dout's W'-dimension
                    i+=1                                        #increment 'i' to advance along dout's H'-dimension
                    j=0                                         #set 'j' to 0 after advancing to a new row along the H'-dimension
        #Remove the padding from dx
        dx=dx[:,:,P:H+P,P:W+P]                                  #padding is there only for H & W dimensions
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        #Get the activation maps' dimensions
        N,C,H,W=x.shape
        #Pooling parameters
        pH,pW,S=pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']
        #Create the pooled activation maps after calculating their H & W dimensions
        Hp=int(((H-pH)/S)+1)               #ignore the fractional part
        Wp=int(((W-pW)/S)+1)               #ignore the fractional part
        aMap=torch.empty((N,C,Hp,Wp),dtype=x.dtype,device=x.device)#pooled activation maps' dimension is (#images,#filters,H',W')
        #Max-pooling begins
        for n in range(N):                 #loop over each image
            for c in range(C):             #loop over each activation map channel
                i=j=0                      #used to index the H & W dimensions of the max-pooked activation maps
                for ht in range(0,H,S):    #loop over the activation map's H-dimension
                    if ht+pH>H:break       #break if the pooling kernel's bottom edge extends beyond x's H-limit
                    for wd in range(0,W,S):#loop over the activation map's W-dimension
                        if wd+pW>W:break   #break if the pooling kernel's right edge extends beyond x's W-limit
                        aMap[n,c,i,j]=torch.max(x[n,c,ht:ht+pH,wd:wd+pW])#max-pooling
                        j+=1               #increment 'j' to advance along x's W-dimension
                    i+=1                   #increment 'i' to advance along x's H-dimension
                    j=0                    #set 'j' to 0 after advancing to a new row along the H-dimension
        out=aMap
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x,params=cache
        #Pooling parameters
        pH,pW,S=params['pool_height'],params['pool_width'],params['stride']
        #Get the dimensions
        N,C,H,W=x.shape
        _,_,Hp,Wp=dout.shape
        #Create the derivative variables. Derivative value is non-zero only for
        #positions where the maximum-value elements were located during
        #max-pooling
        dx=torch.zeros_like(x,device=x.device,dtype=x.dtype)        #same shape as 'x'; don't use 'torch.empty()' for reasons stated previously
        #Backpropagation begins
        for n in range(N):                                          #loop over each image
            for c in range(C):                                      #loop over each filter
                i=j=0                                               #used to index the H & W dimensions of 'dout'
                for ht in range(0,H,S):                             #loop over x's H-dimension
                    if ht+pH>H:break                                #break if the pooling kernel's bottom edge extends beyond x's H-limit
                    for wd in range(0,W,S):                         #loop over x's W-dimension
                        if wd+pW>W:break                            #break if the pooling kernel's right edge extends beyond x's W-limit
                        index=torch.argmax(x[n,c,ht:ht+pH,wd:wd+pW])#position of the maximum value in the region of 'x' that overlaps with the max-pooling kernel's current position
                        dx[n,c,ht+int(index/pW),wd+index%pW]+=dout[n,c,i,j]#convert 'index' to row & column format and write it back to 'dx'
                        j+=1                                        #increment 'j' to advance along dout's W-dimension
                    i+=1                                            #increment 'j' to advance along dout's H-dimension
                    j=0                                             #set 'j' to 0 after advancing to a new row along the H-dimension
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        #Get the I/P dimensions
        C,H,W=input_dims
        #Calculate the dimensions of the network's convolution part's O/P. As
        #the convolution layer's padding and stride preserve the I/P's size, the
        #layer's I/P and O/P both have the same dimensions. Therefore, the FC
        #layer's I/P dimension solely depends on the flattened version of the
        #max-pooling layer's O/P. So, calculate the dimension of the max-pool
        #layer's O/P.
        Hp=int(((input_dims[1]-2)/2)+1)#H'=((H-K)/S)+1; K and S are given in loss( )
        Wp=int(((input_dims[2]-2)/2)+1)#W'=((W-K)/S)+1; K and S are given in loss( )
        #Initialize the network's weights
        self.params['W1']=torch.normal(0.,weight_scale,size=(num_filters,input_dims[0],filter_size,filter_size),dtype=dtype).to(device)
        self.params['b1']=torch.zeros(num_filters,dtype=dtype).to(device)
        self.params['W2']=torch.normal(0.,weight_scale,size=(num_filters*Hp*Wp,hidden_dim),dtype=dtype).to(device)#max-pool layer's O/P dimension is (#filters,Hp,Wp)
        self.params['b2']=torch.zeros(hidden_dim,dtype=dtype).to(device)
        self.params['W3']=torch.normal(0.,weight_scale,size=(hidden_dim,num_classes),dtype=dtype).to(device)
        self.params['b3']=torch.zeros(num_classes,dtype=dtype).to(device)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        #Forward propagation
        A1,cache1=Conv_ReLU_Pool.forward(X,W1,b1,conv_param,pool_param)
        A2,cache2=Linear_ReLU.forward(A1,W2,b2)
        scores,cache3=Linear.forward(A2,W3,b3)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        #Compute the DATA parts of the loss and the gradients
        loss,dZ3=softmax_loss(scores,y)                              #L and ∂L/∂y=∂Z3=∂A3; Z3=A3 since there is no activation for layer 3
        dA2,grads['W3'],grads['b3']=Linear.backward(dZ3,cache3)      #∂L/∂W3 and ∂L/∂b3
        dA1,grads['W2'],grads['b2']=Linear_ReLU.backward(dA2,cache2) #∂L/∂W2 and ∂L/∂b2
        _,grads['W1'],grads['b1']=Conv_ReLU_Pool.backward(dA1,cache1)#∂L/∂W1 and ∂L/∂b1
        #Compute the REG parts of the loss and the gradients
        loss+=(self.reg*(torch.sum(W1*W1)+torch.sum(W2*W2)+torch.sum(W3*W3)))#add the REG part
        grads['W1']+=(2*self.reg*W1)                                 #Add the REG part, which is ∇W1(REG)
        grads['W2']+=(2*self.reg*W2)                                 #Add the REG part, which is ∇W2(REG)
        grads['W3']+=(2*self.reg*W3)                                 #Add the REG part, which is ∇W3(REG)
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        #Get the I/P dimensions
        C,H,W=input_dims
        #Convolution and pooling parameters
        K,P=3,1                                 #kernel size and stride for the convolution layers
        pH=pW=pS=2                              #kernel size and stride for the pooling layers
        #Initialize the network's weights. As the convolution layers preserve
        #dimensions, the I/P and the O/P dimensions will be the same for every
        #convolution layer. However, the application of max-pooling will change
        #the activation map O/P's dimension. Therefore, the final FC layer's I/P
        #will depend on the #max-pooling operations (given in 'max_pools').
        for lyr in range(1,self.num_layers+1):  #loop over each layer
            if lyr==self.num_layers:            #weight initialization for the final FC layer
                if weight_scale=='kaiming':     #use Kaiming weight initialization for W
                    self.params['W'+str(lyr)]=kaiming_initializer(C*H*W,num_classes,K=None,relu=False,device=device,dtype=dtype)
                else:
                    self.params['W'+str(lyr)]=torch.normal(0.,weight_scale,size=(C*H*W,num_classes),dtype=dtype).to(device)
                self.params['b'+str(lyr)]=torch.zeros(num_classes,dtype=dtype).to(device)       #no Kaiming initialization for the bias
            else:                               #weight initialization for all the convolution layers
                if weight_scale=='kaiming':     #use Kaiming weight initialization for W
                    self.params['W'+str(lyr)]=kaiming_initializer(C,num_filters[lyr-1],K=K,relu=True,device=device,dtype=dtype)
                else:
                    self.params['W'+str(lyr)]=torch.normal(0.,weight_scale,size=(num_filters[lyr-1],C,K,K),dtype=dtype).to(device)
                self.params['b'+str(lyr)]=torch.zeros(num_filters[lyr-1],dtype=dtype).to(device)#no Kaiming initialization for the bias
                if self.batchnorm:              #if 'batchnorm' is set to True, initialize γ and β
                    self.params['gamma'+str(lyr)]=torch.ones(num_filters[lyr-1],dtype=dtype).to(device)
                    self.params['beta'+str(lyr)]=torch.zeros(num_filters[lyr-1],dtype=dtype).to(device)
                C=num_filters[lyr-1]            ##channels of the I/P to the next layer is the current layer's #filters
                if (lyr-1) in max_pools:        #update the activation maps' H & W dimensions if the current convolution stage employs a max-pooling layer
                    H=int(((H-pH)/pS)+1)        #the activation maps' H-dimension after the max-pooling operation in the current convolution stage; needed to calculate the final FC layer's I/P dimension
                    W=int(((W-pW)/pS)+1)        #the activation maps' W-dimension after the max-pooling operation in the current convolution stage; needed to calculate the final FC layer's I/P dimension
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        layerOP={}                             #dictionary to save all layer O/Ps
        #Forward propagation
        for lyr in range(1,self.num_layers+1): #loop over each layer
            if lyr==1:                         #forward propagate through the first convolution stage
                if (lyr-1) in self.max_pools:  #the current convolution stage employs a max-pooling layer
                    if self.batchnorm:         #the current convolution stage employs batch-normalization
                        layerOP['A1'],layerOP['cache1']=Conv_BatchNorm_ReLU_Pool.forward(X,self.params['W1'],self.params['b1'],self.params['gamma1'],self.params['beta1'],conv_param,self.bn_params[0],pool_param)
                    else:                      #the current convolution stage does not employ batch-normalization
                        layerOP['A1'],layerOP['cache1']=Conv_ReLU_Pool.forward(X,self.params['W1'],self.params['b1'],conv_param,pool_param)
                else:                          #the current convolution stage does not employ a max-pooling layer
                    if self.batchnorm:         #the current convolution stage employs batch-normalization
                        layerOP['A1'],layerOP['cache1']=Conv_BatchNorm_ReLU.forward(X,self.params['W1'],self.params['b1'],self.params['gamma1'],self.params['beta1'],conv_param,self.bn_params[0])
                    else:
                        layerOP['A1'],layerOP['cache1']=Conv_ReLU.forward(X,self.params['W1'],self.params['b1'],conv_param)
                continue
            elif lyr==self.num_layers:         #the final layer is a FC layer. So max-pooling and batch-normalization will not be employed
                scores,layerOP['cache'+str(lyr)]=Linear.forward(layerOP['A'+str(lyr-1)],self.params['W'+str(lyr)],self.params['b'+str(lyr)])
                continue
            else:                              #forward propagate through the intermediate convolution stages
                if (lyr-1) in self.max_pools:  #the current convolution stage employs a max-pooling layer
                    if self.batchnorm:         #the current convolution stage employs batch-normalization
                        layerOP['A'+str(lyr)],layerOP['cache'+str(lyr)]=Conv_BatchNorm_ReLU_Pool.forward(layerOP['A'+str(lyr-1)],self.params['W'+str(lyr)],self.params['b'+str(lyr)],self.params['gamma'+str(lyr)],self.params['beta'+str(lyr)],conv_param,self.bn_params[lyr-1],pool_param)
                    else:                      #the current convolution stage does not employ batch-normalization
                        layerOP['A'+str(lyr)],layerOP['cache'+str(lyr)]=Conv_ReLU_Pool.forward(layerOP['A'+str(lyr-1)],self.params['W'+str(lyr)],self.params['b'+str(lyr)],conv_param,pool_param)
                else:                          #the current convolution stage does not employ a max-pooling layer
                    if self.batchnorm:         #the current convolution stage employs batch-normalization
                        layerOP['A'+str(lyr)],layerOP['cache'+str(lyr)]=Conv_BatchNorm_ReLU.forward(layerOP['A'+str(lyr-1)],self.params['W'+str(lyr)],self.params['b'+str(lyr)],self.params['gamma'+str(lyr)],self.params['beta'+str(lyr)],conv_param,self.bn_params[lyr-1])
                    else:                      #the current convolution stage does not employ batch-normalization
                        layerOP['A'+str(lyr)],layerOP['cache'+str(lyr)]=Conv_ReLU.forward(layerOP['A'+str(lyr-1)],self.params['W'+str(lyr)],self.params['b'+str(lyr)],conv_param)
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        #Compute the DATA parts of the loss and the gradients
        layerGrad={}                           #dictionary to save all activation gradients
        #Back propagation - compute the DATA parts of the loss and the gradients
        for lyr in range(self.num_layers,0,-1):#loop over each layer
            if lyr==self.num_layers:           #back propagate through the last layer, which is a FC layer
                loss,dZL=softmax_loss(scores,y)#L and ∂L/∂y=∂ZL=∂AL; ZL=AL since there is no activation for the Lth layer (or the final) layer
                layerGrad['dA'+str(lyr-1)],grads['W'+str(lyr)],grads['b'+str(lyr)]=Linear.backward(dZL,layerOP['cache'+str(lyr)])
                continue
            else:                              #back propagate through the convolution stages
                if (lyr-1) in self.max_pools:  #the current convolution stage employs a max-pooling layer
                    if self.batchnorm:         #the current convolution stage employs batch-normalization
                        layerGrad['dA'+str(lyr-1)],grads['W'+str(lyr)],grads['b'+str(lyr)],grads['gamma'+str(lyr)],grads['beta'+str(lyr)]=Conv_BatchNorm_ReLU_Pool.backward(layerGrad['dA'+str(lyr)],layerOP['cache'+str(lyr)])#∂L/∂Wlyr, ∂L/∂blyr, ∂L/∂γlyr, and ∂L/∂βlyr
                    else:                      #the current convolution stage does not employ batch-normalization
                        layerGrad['dA'+str(lyr-1)],grads['W'+str(lyr)],grads['b'+str(lyr)]=Conv_ReLU_Pool.backward(layerGrad['dA'+str(lyr)],layerOP['cache'+str(lyr)])#∂L/∂Wlyr and ∂L/∂blyr
                else:                          #the current convolution stage does not employ a max-pooling layer
                    if self.batchnorm:         #the current convolution stage employs batch-normalization
                        layerGrad['dA'+str(lyr-1)],grads['W'+str(lyr)],grads['b'+str(lyr)],grads['gamma'+str(lyr)],grads['beta'+str(lyr)]=Conv_BatchNorm_ReLU.backward(layerGrad['dA'+str(lyr)],layerOP['cache'+str(lyr)])     #∂L/∂Wlyr, ∂L/∂blyr, ∂L/∂γlyr, and ∂L/∂βlyr
                    else:                      #the current convolution stage does not employ batch-normalization
                        layerGrad['dA'+str(lyr-1)],grads['W'+str(lyr)],grads['b'+str(lyr)]=Conv_ReLU.backward(layerGrad['dA'+str(lyr)],layerOP['cache'+str(lyr)])     #∂L/∂Wlyr and ∂L/∂blyr
        #Back propagation - compute the REG parts of the loss and the gradients
        temp=0.
        for lyr in range(1,self.num_layers+1): #loop over each layer
            temp+=torch.sum(self.params['W'+str(lyr)]*self.params['W'+str(lyr)])#accumulates all the REG parts
            grads['W'+str(lyr)]+=(2*self.reg*self.params['W'+str(lyr)])         #add the REG part, which is ∇Wlyr(REG)
        loss+=(self.reg*temp)
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    #Choices of α and initial weight distribution's σ needed to overfit the
    #training set
    learning_rate,weight_scale=5e-3,5e-2
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    #Define the model
    """
    model=DeepConvNet(input_dims=data_dict['X_train'].shape[1:],num_classes=10,
                      reg=1e-4,num_filters=[8,16,32,64,128],
                      max_pools=[0,1,2,3,4],weight_scale='kaiming',
                      dtype=dtype,device=device)                               
    #Define the solver
    solver=Solver(model,data_dict,optim_config={'learning_rate':0.09},
                  batch_size=128,num_epochs=5,update_rule='adam',device=device)
    """
    model=DeepConvNet(input_dims=data_dict['X_train'].shape[1:],num_classes=10,
                      reg=5e-4,num_filters=[16,32,64,128],max_pools=[0,1,2,3],
                      weight_scale='kaiming',dtype=dtype,device=device)        #as the dataset is CIFAR10, #O/P classes is 10
    solver=Solver(model,data_dict,optim_config={'learning_rate':2e-3},
                  batch_size=128,num_epochs=10,update_rule=adam,device=device)
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        #Initialize the weights for a FC layer
        weight=((gain/Din)**0.5)*torch.normal(0.,1,size=(Din,Dout),dtype=dtype,device=device)
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        #Initialize the weights for a convolution layer
        weight=((gain/(Din*K*K))**0.5)*torch.normal(0.,1,size=(Dout,Din,K,K),dtype=dtype,device=device)
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            #Forward propagation
            mu=torch.mean(x,dim=0)                                  #the mini-batch means of all features i.e., along the row dimension. Output shape is (D,)
            #sigma2=torch.var(x,dim=0)                               #the mini-batch variances of all features i.e., along the column dimension. Output shape is (D,)
            sigma2=torch.sum((x-mu)**2,dim=0)/N                     #the mini-batch variances of all features i.e., along the column dimension. Output shape is (D,)
            xHat=(x-mu)/torch.sqrt(sigma2+eps)                      #normalizing the mini-batch of data using μ and σ2
            #Scale and shift the O/P label
            out=(gamma*xHat)+beta                                   #yMB=(γ*xHat)+β; 'MB' is the current mini-batch
            #Updatethe running averages of the means and the variances
            running_mean=(momentum*running_mean)+((1-momentum)*mu)
            running_var=(momentum*running_var)+((1-momentum)*sigma2)
            #Update the cache
            bn_param['running_mean'],bn_param['running_var']=running_mean.detach(),running_var.detach()
            bn_param['eps']=eps
            cache=(gamma,beta,bn_param,x,mu,sigma2,xHat)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            #Back propagation - use the running means and the variances
            #calculated during training to normalize a mini-batch of test data
            xHat=(x-running_mean)/torch.sqrt(running_var+eps)
            #Scale and shift the O/P label
            out=(gamma*xHat)+beta                                   #yMB=(γ*xHat)+β; 'MB' is the current mini-batch
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your 
        #Reference and inspiration - https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        #Get the I/P dimensions
        N,D=dout.shape                                           #'dout' is the same shape as x
        #The parameters of the batch-normalization layer
        gamma,beta,bn_param,x,mu,sigma2,xHat=cache
        eps=bn_param['eps']
        #Backpropagate along the computational graph up until the point (x-μ)/sqrt(σ2+ε).
        #As we process one step at a time, +β comes first, then the product
        #(γ*xHat), and so on.
        dbeta=1.*torch.sum(dout,dim=0)                           #computation of ∂L/∂β is similar to that of the bias in a convolution layer i.e., get the sum of all derivatives across all samples as each sample contributes to learning β (β is common to all samples)
        dgamma=torch.sum((dout*xHat),dim=0)                      #computation of ∂L/∂γ is similar to that of the weights in a convolution layer. After multiplying 'dout' with 'xHat', get the sum over all samples like above as as each sample contributes to learning γ (γ is common to all samples)
        dxHat=dout*gamma                                         #same idea as above as both γ and xHat are involved with the same product operation. However, since here we deal with xHat for which we we must look at individual samples, we do not sum the derivative over all samples n xHat
        dNum=(torch.sqrt(sigma2+eps)**-1)*dxHat                  #computation of ∂L/∂(x-μ)2 is similar to that of ∂L/∂xHat        
        dDen=torch.sum(dxHat*(x-mu),dim=0)                       #computation of ∂L/∂(sqrt(σ2+ε)^-1) is similar to that of ∂L/∂γ i.e., get the sum of all derivatives across all samples
        #From here on, backpropagation takes two paths - the calculation of the
        #numerator part (x-μ) and the calculation of the denomenator part
        #1/sqrt(σ2+ε). Along each path, the computation is done one step at a 
        #time
        dsqrtSigma2=(-1/torch.pow(torch.sqrt(sigma2+eps),2))*dDen#bacpropagate along the path that calculates the variance. Note that the derivative of an fraction 1/f is -1/f^2
        dsigma2=(0.5/torch.sqrt(sigma2+eps))*dsqrtSigma2         #derivative of sqrt(f) is 0.5*(sqrt(f)^-1)
        dSum=(1/N)*torch.ones_like(dout)*dsigma2                 #derivative of the summation operation: σ2 is a vector of shape (D,) that captures the variances of all features across all samples. So, multiplication with a matrix of shape (N,D) containing all ones will evenl propoagate the variance of each dimension to all samples after following vhe order of dimension
        dSquare=2*(x-mu)*dSum                                    #derivative of (x-μ)^2=2*(x-μ)
        dSub=dNum+dSquare                                        #derivative of the subtraction operation. As gradient signs adjust accordingl, -sign can be replaced with +sign. This is also the dx along the path that directly passes 'x' along
        #From here on, backpropagation again takes two paths - the calculation
        #of μ and the path thas passes along x
        dxP1=dSub                                                #derivative along the first path is equal to dSub
        dmu=-1.*torch.sum(dxP1,dim=0)                            #derivative along the path that calculates (x-μ). Derivative of that is -μ. And just like the bias term, get the sum of all derivatives across all samples as each sample contributes to learning μ (μ is common to all samples)
        dxP2=(1/N)*(torch.ones_like(dout))*dmu                   #just like the calculation of the term 'dSum'
        dx=dxP1+dxP2                                             #the final derivative ∂L/∂x        
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        #Get the I/P dimensions
        N,D=dout.shape                                #'dout' is the same shape as x
        #The parameters of the batch-normalization layer
        gamma,beta,bn_param,x,mu,sigma2,xHat=cache
        eps=bn_param['eps']
        #Backpropagation
        dbeta=torch.sum(dout,dim=0)
        dgamma=torch.sum((dout*xHat),dim=0)
        dx=prod=dout*gamma                            #stage 1 backpass
        dx-=torch.sum(prod,dim=0)/N                   #stage 2 backpass
        dx-=(torch.sum(prod*xHat,dim=0)*xHat)/N       #stage 3 backpass
        dx/=torch.sqrt(sigma2)
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        #Forward propagation
        #Get the I/P dimensions
        N,C,H,W=x.shape 
        #Convert the I/P to a form that is appropriate for batch-normalization
        xClone=x.clone()
        xClone=torch.permute(xClone,(0,2,3,1))                 #as given in the description, since the mean and the variance are computed separately for each channel and BN works on an I/P of dimension (N,D) we rearrange the dimensions to the format (N,H,W,C)
        xClone=xClone.reshape(N*H*W,C)                         #convert 'xClone' to a dimension that is appropriate for BN (which is (N*H*W,C))
        #Forward propagation
        xBN,cache=BatchNorm.forward(xClone,gamma,beta,bn_param)#batch-normalize the I/P
        #Convert the batch-normalized version of 'x' to the shape (N,C,H,W)
        xBN=xBN.reshape(N,H,W,C)                               #'xBN' now has the shape (N,H,W,C)
        out=torch.permute(xBN,(0,3,1,2))                       #'out' now has the shape (N,C,H,W)
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        #Get the I/P dimensions
        N,C,H,W=dout.shape                    #'dout' is the same shape as x
        #Convert the I/P to a form that is appropriate for batch-normalization
        dClone=dout.clone()
        dClone=torch.permute(dClone,(0,2,3,1))#as in forward propagation, as the mean and the variance are computed separately for each channel and BN works on an I/P of dimension (N,D) we rearrange the dimensions to the format (N,H,W,C)
        dClone=dClone.reshape(N*H*W,C)        #convert 'dClone' to a dimension appropriate for BN (which is (N*H*W,C))
        #Back propagation       
        dx,dgamma,dbeta=BatchNorm.backward(dClone,cache)
        #Convert the batch-normalized version of 'dx' to the shape (N,C,H,W)
        dx=dx.reshape(N,H,W,C)                #'dx' now has the shape (N,H,W,C)
        dx=torch.permute(dx,(0,3,1,2))        #'dx' now has the shape (N,C,H,W)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
