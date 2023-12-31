## Classification models
We use the following classification models to classify images into different classes: <br /><br />
**Fully-connected deep neural network:** <br />
A deep neural network with fully-connected layers is trained separately using the optimizers - SGD with momentum, RMSProp, and Adam - and dropout regularization. The network uses non-linear ReLU activations in its hidden layers.<br /><br />
**Convolutional neural network (CNN):** <br />
A CNN with VGG-style network architecture - a convolution layer with a 3x3 kernel and a padding size of 1 followed by a max-pooling layer with a 2x2 kernel and a stride of 2 - is trained using Adam optimizer. <br /><br />

## Dataset
The dataset used in this experiment is CIFAR-10, which is a collection of 60,000 3x32x32 images, each belonging to one of ten pre-defined labels: _plane_, _car_, _bird_, _cat_, _deer_, _dog_, _frog_, _horse_, _ship_, and _truck_. <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/1.%20k%20Nearest%20Neighbors/Capture.PNG) <br />

## Results
**Fully-connected deep neural network:** <br />
Training and validation progress corresponding to different weight optimizers for a deep FC neural network with architecture <br />
I/P-FC-ReLU-FC-ReLU-FC-ReLU-FC-ReLU-FC-ReLU-FC-Softmax <br />
and 100 neurons in each of its hidden layers: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/1.%20Fully-conected%20neural%20network/Progress1.png) <br />
Training and validation progress corresponding to different drop-out probabilities for deep FC neural networks with widths of 256 and 512 neurons:<br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/1.%20Fully-conected%20neural%20network/Progress2.png) <br /><br />
**Convolutional neural network (CNN):** <br />
Effect of batch-normalization on the progress of network training and validation: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/2.%20Convolutional%20neural%20network/Progress1.png) <br />
Effect of batch-normalization and learning rate variations on the progress of network training and validation: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/2.%20Convolutional%20neural%20network/Progress2.png) <br /><br />
