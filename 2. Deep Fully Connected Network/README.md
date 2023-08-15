## Classification models
We use the following classification models to classify images into different classes: <br /><br />
**Support Vector Machine (SVM):** <br />
A SVM Classifier will be implemented with a fully-vectorized loss-function, which will be optimized using stochastic gradient descent (SGD). <br /><br />
**Softmax Classifier:** <br />
A Softmax Classifier will be implemented with a softmax loss-function, which will be optimized using SGD. <br /><br />
**Fully-connected deep neural network:** <br />

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
![]() <br />
![]() <br />
![]() <br />
