#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./visualization.jpg "Visualization"
[image2]: ./grayscale.jpg "Grayscaling"
[image3]: ./random_noise.jpg "Random Noise"
[image4]: ./test1.jpg "Traffic Sign 1"
[image5]: ./test2.jpg "Traffic Sign 2"
[image6]: ./test3.jpg "Traffic Sign 3"
[image7]: ./test4.jpg "Traffic Sign 4"
[image8]: ./test5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gongdn/CarND-Traffic-Sign-Classifier-Project-master)


###Data Set Summary & Exploration
####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
The code for this step is contained in the [2] code cell of the IPython notebook.  
I used the pandas library to calculate summary statistics of the traffic
signs data set:
Number of training examples = 34799
Number of validdation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.
The code for this step is contained in the [165] code cell of the IPython notebook.  
After executing the cell, it shows an exploratory visualization of the data set. It is a bar chart showing the count and luminance characteristics of each class of signs (min/max/mean). Also it plots randomly picked eight traffic sign images from each classe of signs. 


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to do color to gray coversion 

Here is an example of a traffic sign image before and after grayscaling.
Before gray scaling
![alt text][train_ori/class_0_9960.jpg]
After gray saling
![alt text][train_norm/class_0_9960.jpg]

As a last step, I normalized the image data because zero mean is helpful for later training. The average of the original data set is about -0.35 after using a/127.5-1.0. So the simple linear normailize is not zero mean. Finally I normialized the image from [0,255] to [-1,1] by using historgram normalization. The mean value for data set is very close to 0 now.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code to add new augmented images are in code cell [50].
Add 820 samples for 0 Speed limit (20km/h) class
Add 640 samples for 6 End of speed limit (80km/h) class
Add 310 samples for 14 Stop class
Add 460 samples for 15 No vehicles class
Add 640 samples for 16 Vehicles over 3.5 metric tons prohibited class
Add 10 samples for 17 No entry class
Add 820 samples for 19 Dangerous curve to the left class
Add 700 samples for 20 Dangerous curve to the right class
Add 730 samples for 21 Double curve class
Add 670 samples for 22 Bumpy road class
Add 550 samples for 23 Slippery road class
Add 760 samples for 24 Road narrows on the right class
Add 460 samples for 26 Traffic signals class
Add 790 samples for 27 Pedestrians class
Add 520 samples for 28 Children crossing class
Add 760 samples for 29 Bicycles crossing class
Add 610 samples for 30 Beware of ice/snow class
Add 310 samples for 31 Wild animals crossing class
Add 790 samples for 32 End of all speed and passing limits class
Add 401 samples for 33 Turn right ahead class
Add 640 samples for 34 Turn left ahead class
Add 670 samples for 36 Go straight or right class
Add 820 samples for 37 Go straight or left class
Add 730 samples for 39 Keep left class
Add 700 samples for 40 Roundabout mandatory class
Add 790 samples for 41 End of no passing class
Add 790 samples for 42 End of no passing by vehicles over 3.5 metric tons class
Augmented Train Data Set Size,Max,Min,Mean: 51690 1.0 -1.0 0.0805464956692

After the executing the cell, it will show the augmented image in gray. The size of traing set is increased from 34799 to 51690.
The mean is increased from 0.03 to 0.08. 

Each class will have 1000 training samples with extra augmentation. 
I used the specification for augmentation as follows:
aug_spec={"brightness": randomly change gray value with scalar factor range of [0.9,1.1], # 
         "angle":5, randomly change angle with range of [-2.5 degree to 2.5 degree]
         "tran":4, randomly change offset in the range of [-2,2] and [-2,2] in Horinzontal and vertical direction
         "scale":[0.75,1.5]} randomly scale up or down in the range of [0.75, 1.5]
          }
Also flip with a probability of 25% for specific class, excluding classes with numbers, words and left/right arrows.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the [123] cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 				|
| Convolution 3x3	    | 1x1 stride, valid padding, output 10x10x16.      									|
| RELU | |
| Max pooling | 2x2 stride, outputs 5x5x16 |
| Fully connected		| inputs 400, outputs 128        									|
| RELU | |
| drop out | keep prob = 0.8]
| Fully connected  | inputs 128, outputs 256                 | 
| RELU | |
| drop out | keep prob = 0.8]
| Fully connected  | inputs 256, outputs 43                 |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
