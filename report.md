# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/data_visualization.png "Visualization"
[image2]: ./report/data_augmentation.png "Data augmentation"
[image3]: ./report/pre-processing.png "Pre-processing"
[image4]: ./report/AlexNet-Loss_and_accuracies.png "AlexNet Loss and accuracies plot"
[image5]: ./report/New_images.png "New images"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

This is a link to my [project code](https://github.com/sebastien-attia/Udacity__CarND-TrafficSignClassifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Load of the data.

The training, validation and testing sets can be downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads).

The code for this step is contained in the 1st and 2nd code cells (Step 0 and Step 1) of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34 799 images.
* The size of test set is 12 630 images.
* The shape of a traffic sign image is 32x32x3 (RGB image).
* The number of unique classes/labels in the data set is 43 different signs.

#### 2. Exploratory visualization of the dataset.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.

The following shows a random image for each of the 43 different signs.

![Data visualization][image1]

### Design and Test a Model Architecture

#### 1. The datasets and the data augmentation

The dataset is split in 3 different dataset:
- 1 dataset for training (34 799 images),
- 1 dataset for the validation (4 410 images),
- 1 dataset for testing (generalization) (12 630 images).

The 4th code cell of the IPython notebook contains the code for generating a new dataset from the training dataset. I decided to generate additional data to see the impact of the size of the dataset on the accuracy of the model.
To add more data to the the data set, I used the following techniques:
- rotation of the image with a random angle between -15° and 15°,
- shift of the image with a random number between -3 and 3 pixels,
- a combination of rotation and shift.
The transformation applied is choosen randomly.

Here is an example of an original image and an augmented image:

![Data augmentation][image2]

#### 2. Pre-processing.

The code for this step is contained in the 5th code cell of the IPython notebook.

A new dataset is created from the training set, the following transformations are applied:
- convert the image to grayscale,
- normalize the image,

The goal of this new dataset is to see if the color has an impact positive or
negative on the accuracy.

![Pre-processing][image3]


#### 3. Model architecture

The code for my final model is located in the 6th and 8th cells of the IPython notebook.


My final model consisted of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							            |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	  |
| RELU					        |												                        |
| Max pooling	3x3      	| 2x2 stride, same padding, outputs 16x16x64 		|
| Normalize         	  |                                           		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	  |
| RELU					        |												                        |
| Normalize         	  |                                           		|
| Max pooling	      	  | 2x2 stride, same padding, outputs 8x8x64 		  |
| Fully connected		    | inputs 4096, outputs 384        				      |
| Fully connected		    | inputs 384,  outputs 192        				      |
| Fully connected		    | inputs 192,  outputs 43        				        |
| Softmax				        | inputs 43,   outputs 43        						 	  |


The first architecture I have chosen was LeNet. I got pretty good results, up to an
accuracy of 93.6% on the test dataset (trained on the augmented dataset).
But I wanted to see if I could have better results. So I tried with a simplified
version of AlexNet, a proven architecture. I got better results.

But I wanted to check if I could have better results with a "superloaded" version
of LeNet. So I created the LeNet "superloaded"'s version. I got a slightly better
result (test accuracy) than the LeNet, but a little bit less than the AlexNet
architecture.

I have tried too the momentum optimizer for the backpropagation. But I had difficulties
to make it converged. As each experimentation is time consuming, I kept the Adam
optimizer, easier to setup.


#### 4. Training of the model

For the following part of the project, I have chosen to use AlexNet because the
test accuracy is the best.

The code for training the model is located in the 15th cell of the IPython notebook.

To train the model, I used the following hyperparameters:
- the augmented training set,
- the Adam optimizer for the backpropagation with a learning rate of 0.001,
- no drop-out or regularization term,
- a batch size of 256, and
- a EPOCH of 26

It takes 3 hours to train the model on a K80 GPU.

![AlexNet - Loss and training/validation accuracies][image4]

I suppose we could have better results by increasing the batch size to limit the "noise"
in the loss.


#### 5. Analysis of the results

The code for calculating the accuracy of the model is located in the 15th cell of the IPython notebook.

My final model results were:
* training set accuracy of 99 %.
* validation set accuracy of 96 %.
* test set accuracy of 94.8 %


I have done several experimentations with the 3 different models:
- LeNet,
- AlexNet (simplified),
- LeNet superloaded.

And I get the following results:

| Model     |  Train. acc. | Val. acc.    |  Test acc.   |
|:---------:|:------------:|:------------:|:------------:|
| AlexNet   |    99%   		 |   96%        | 94.8%        |
| LeNet(superloaded)| 98% | 95% | 93.7 % |  
| LeNet | 99% | 94% | 93.6 % |  


### Test a Model on New Images

#### 1. 10 German traffic signs found on the web

Here are 10 German traffic signs that I found on the web:

![New images][image5]

The first and third images might be difficult to classify because the images are dark.
The second image might be difficult to classify because the sign overlap another
similar sign and is a little bit rotated.
The sixth image might be difficult to classify because the image has no contrast.


#### 2. The model's predictions

The code for making predictions on my final model is located in the 30th cell of the IPython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Priority road  									|
| Road work     			| Speed limit (50km/h) 										|
| Speed limit (100km/h)	| Speed limit (100km/h)											|
| No passing	      		| No passing					 				|
| Bumpy road			| Bumpy road     							|
| Speed limit (70km/h)			| Speed limit (70km/h)     							|
| Yield      		| Yield 									|
| Traffic signals     			| Traffic signals 										|
| Ahead only	| Ahead only											|
| Speed limit (50km/h)	      		| Speed limit (50km/h)					 				|



The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 94.8%.

#### 3. How certain the model is when predicting ?

The code for making predictions on my final model is located in the 32nd cell of the IPython notebook.

The top 10 soft max probabilities are:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Priority road    									|
| .43     				| Speed limit (50km/h)										|
| .97					| Speed limit (100km/h)											|
| .99	      			| No passing					 				|
| .99				    | Bumpy road       							|
| .99				    | Speed limit (70km/h)      							|
| .99				    | Yield     							|
| .99				    | Traffic signals      							|
| .70				    | Ahead only      							|
| .99				    | Speed limit (50km/h)      							|

Apart for the second image, the model is very certain when predicting (from 70% to 100%).
For the second image, the model has no certainty and make a wrong prediction.
