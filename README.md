# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup_images/Image_Visualization.png "Visualization"
[image2]: ./writeup_images/bar_train.png "Bar Graph Training data set"
[image3]: ./writeup_images/bar_test.png "Bar Graph Testing data set"
[image4]: ./writeup_images/bar_valid.png "Bar Graph Validation data set"
[image5]: ./writeup_images/befora_after_gray.png "Before and after gray scale"
[image6]: ./writeup_images/Aug_images.png "Augmented Images"
[image7]: ./writeup_images/new_images.png "New Images"
[image8]: ./writeup_images/results.png "results Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
1. I have ensured that i have submitted all the  files required : the notebook that contains the code , the html file and the writeup markdown file .


2. For the summary of the data set i have calculated the number of the training , testing and validation examples and the shape of the image which was (32,32,3) and for the exploratory visualization i have showed some of the images provided in the given data set and i have plotted a bar graph to display the distribution of each label in each training , testing and validation data sets .


3. For the data preprocessing i have applied several techniques which i have read on the literature the techniques included : transforming the images from rgb to gray scale sothat i have only one channel which will decrease the time consumption ,Normalizing to reduce the values of each pixels in the images .Also i have  random augmentation by tilting the images and then adding those images to the training and validation data sets so that i improve the accuracy of my model .and for data training i used adam optimizier as it gave better results for me than other optimizers and for the epochs i started with 10 and the results wasn't good then i choosed 30 and the model was overfitting so it reduced it to 24 which was better and gave a validation accuracy of almost 97% , for the batch size i choosed a 156 batch size and a learning rate of 0.00097 which i found while i was reading the literature .



4. For testing the  model on new images, I got 5  new images on the internet which i can find their label in the given 43 labels and then i rescaled the images to be 32x32x3 so that i can apply LeNet model on it and then i converted the images from rgb to grayscale and i have normalized the data of the new images and then i applied my model on it and then i showed the softmax probabilities for each image for the whole 43 labels.


---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and numpy  libraries  to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:

first i will display some random images which was provied in the data set

![alt text][image1]


then i have plotted a bar graph to represent the distribution of labels in each data set :

For training data set :

![alt text][image2]


For testing data set :

![alt text][image3]

For Validation data set :

![alt text][image4]




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale sothat the images will have only one channel instead of 3 to reduce the time consumption for the model and then i have normalized the data to reduce tha values of the pixels in the images .

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]


I decided to generate additional and i have added this data to the original data set to prevent the model from overfitting and to increase the accuracy of the model to achieve almost 97% validation accuracy .

To add more data to the the data set, I have made random augmentation by tilting the images and then adding those images to the training and validation data sets so that i improve the accuracy of my model

Here is an example of an original image and an augmented image:
![alt text][image6]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet architecure beside adding an additional convloution layer without using maxpooling and then i used AdamOptimizer with a learning rate of 0.0097 which i found it usefull i used 24 epochs with 156 batch size i was able to reach a validation accuracy of almost 97% which is above the baseline 93% .
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 97.6%
* test set accuracy of 94.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 *  I used a  similar architecture to the paper offered by the instructors because of the good results this paper shows.
* What were some problems with the initial architecture?
* lake of the provided data and the validation accuracy wasn't good enough.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Adding another convolution layer and augmenting some more data .
* Which parameters were tuned? How were they adjusted and why?
* Epoch, learning rate, batch size, and drop out probability .  For the number of epochs  the main reason I tuned this was after I started to get better accuracy i started with 10 then moved to 30 then i got a better resulst when setting the epochs to 24 .  The batch size I increased only slightly since starting once I increased the dataset size.  The learning rate i started with 0.001 which was the default in many papers then i adjusted it to be 0.0097 which gave better results .
* What are some of the important design choices and why were they chosen?
* from my point of view i think that it's important to get more uniform dataset with enough conv layers so that the accuracy will improve .

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 30 km/h     	     	| 30 km/h  									    |
| Bumpy Road     		| Bumpy Road 									|
| Ahead Only		    | Ahead Only									|
| No vehicles	      	| No vehicles					 				|
|Go straight or left	| Go straight or left      						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image8]
