## Classification models
We use the following classification models to classify images into different classes: <br /><br />
**Fully-connected deep neural network:** <br />
A deep neural network with fully-connected layers is trained separately using the optimizers - SGD with momentum, RMSProp, and Adam - and dropout regularization. The network uses non-linear ReLU activations in its hidden layers.<br /><br />
**Convolutional neural network (CNN):** <br />
<br /><br />

## Dataset
The dataset used in this experiment is CIFAR-10, which is a collection of 60,000 3x32x32 images, each belonging to one of ten pre-defined labels: _plane_, _car_, _bird_, _cat_, _deer_, _dog_, _frog_, _horse_, _ship_, and _truck_. <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/1.%20k%20Nearest%20Neighbors/Capture.PNG) <br />

## Results
**Fully-connected deep neural network:** <br />
Training and validation progress corresponding to different weight optimizers for a deep FC neural network with architecture <br />
I/P-FC-ReLU-FC-ReLU-FC-ReLU-FC-ReLU-FC-ReLU-FC-Softmax <br />
and 100 neurons in each of its hidden layers: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/1.%20Fully-conected%20neural%20network/Progress1.png) <br />
Training and validation progress corresponding to different drop-out probabilities for a deep FC neural network with 20 neurons in its first hidden layer and 30 neurons in its second:<br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/1.%20Fully-conected%20neural%20network/Progress2.png) <br />
![]() <br />
Training progress and final weights for the best model identified through cross-validation (α=1e-1, λ=0, and a hidden layer size of 512): <br />
![]() <br />
![]() <br />
![]() <br />
**Convolutional neural network (CNN):** <br />
![]() <br />
![]() <br />
![]() <br /><br />
