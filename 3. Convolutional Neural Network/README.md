## Classification models
We use the following classification models to classify images into different classes: <br /><br />
**Fully-connected deep neural network:** <br />
A deep neural network with fully-connected layers is trained separately using the optimizers - RMSProp and Adam - and dropout regularization. The network uses non-linear ReLU activations in its hidden layer, and its architecture is as follows: <br /><br />
I/P-
**Convolutional neural network (CNN):** <br />
<br /><br />

## Dataset
The dataset used in this experiment is CIFAR-10, which is a collection of 60,000 3x32x32 images, each belonging to one of ten pre-defined labels: _plane_, _car_, _bird_, _cat_, _deer_, _dog_, _frog_, _horse_, _ship_, and _truck_. <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/1.%20k%20Nearest%20Neighbors/Capture.PNG) <br />

## Results
**Fully-connected deep neural network:** <br />
Training progress for hidden layer sizes of 2, 8, 32, and 128 when α=1e-1 and λ=1e-3: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/TrainAccuracy1.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/ValAccuracy1.png) <br />
Training progress for λ=0, 1e-5, 1e-3, and 1e-1 when α=1e-1 and the hidden layer size is 128: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/TrainAccuracy2.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/ValAccuracy2.png) <br />
Training progress for α=1e-4, 1e-2, 1e0, and 1e2 when λ=1e-4 and the hidden layer size is 128: <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/TrainAccuracy3.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/ValAccuracy3.png) <br />
Training progress and final weights for the best model identified through cross-validation (α=1e-1, λ=0, and a hidden layer size of 512): <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/LossHistory.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/TrainAccuracyHistory.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/FinalWeights.png) <br />
**Convolutional neural network (CNN):** <br />
![]() <br />
![]() <br />
![]() <br /><br />
