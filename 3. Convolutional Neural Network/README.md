## Classification models
We use the following classification models to classify images into different classes: <br /><br />
**Fully-connected deep neural network:** <br />
A deep neural network with fully-connected layers is trained separately using the optimizers - SGD with momentum, RMSProp, and Adam - and dropout regularization. The network uses non-linear ReLU activations in its hidden layer, and its architecture is: <br />
I/P-FC-ReLU-FC-ReLU-FC-ReLU-FC-ReLU-FC-ReLU-FC-Softmax <br />
Each FC hidden layer has 100 neurons in it.<br /><br />
**Convolutional neural network (CNN):** <br />
<br /><br />

## Dataset
The dataset used in this experiment is CIFAR-10, which is a collection of 60,000 3x32x32 images, each belonging to one of ten pre-defined labels: _plane_, _car_, _bird_, _cat_, _deer_, _dog_, _frog_, _horse_, _ship_, and _truck_. <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/1.%20k%20Nearest%20Neighbors/Capture.PNG) <br />

## Results
**Fully-connected deep neural network:** <br />
Training and validation progress for different weight optimizers: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/3.%20Convolutional%20Neural%20Network/Results/1.%20Fully-conected%20neural%20network/Progress1.png) <br />
Training progress for α=1e-4, 1e-2, 1e0, and 1e2 when λ=1e-4 and the hidden layer size is 128: <br />
![]() <br />
![]() <br />
Training progress and final weights for the best model identified through cross-validation (α=1e-1, λ=0, and a hidden layer size of 512): <br />
![]() <br />
![]() <br />
![]() <br />
**Convolutional neural network (CNN):** <br />
![]() <br />
![]() <br />
![]() <br /><br />
