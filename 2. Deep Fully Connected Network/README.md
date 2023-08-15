## Classification models
We use the following classification models to classify images into different classes: <br /><br />
**Support Vector Machine (SVM):** <br />
A SVM Classifier will be implemented with a fully-vectorized loss-function, which will be optimized using stochastic gradient descent (SGD). <br /><br />
**Softmax Classifier:** <br />
A Softmax Classifier will be implemented with a softmax loss-function, which will be optimized using SGD. <br /><br />
**Fully-connected deep neural network:** <br />
A deep neural network with fully-connected layers is trained using a Softmax loss-function and an L<sub>2</sub> regularizer. The network uses a non-linear ReLU activation in its hidden layer. The network's architecture is: <br />
I/P-FC-ReLU-FC-Softmax <br /><br />

All the above classification models will unravel an image into a row vector before classifying it, resulting in a loss of spatial information and low accuracies. However, the accuracies will be higher than what resulted from the use of the _k_ nearest neighbors algorithm. <br /><br />

## Dataset
The dataset used in this experiment is CIFAR-10, which is a collection of 60,000 3x32x32 images, each belonging to one of ten pre-defined labels: _plane_, _car_, _bird_, _cat_, _deer_, _dog_, _frog_, _horse_, _ship_, and _truck_. <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/1.%20k%20Nearest%20Neighbors/Capture.PNG) <br />

## Results
**Support Vector Machine (SVM):** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/1.%20SVM/TrainAccuracy.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/1.%20SVM/ValAccuracy.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/1.%20SVM/FinalWeights.png) <br /><br />
**Softmax Classifier:** <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/2.%20Softmax/TrainAccuracy.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/2.%20Softmax/ValAccuracy.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/2.%20Softmax/FinalWeights.png) <br /><br />
**Fully-connected deep neural network:** <br />
Training progress for hidden layer sizes of 2, 8, 32, and 128 when α=0.1 and =0.001
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/TrainAccuracy1.png) <br />
![](https://github.com/rprasan/Computer-Vision/blob/main/2.%20Deep%20Fully%20Connected%20Network/Results/3.%20DNN/ValAccuracy1.png) <br />
![]() <br />
