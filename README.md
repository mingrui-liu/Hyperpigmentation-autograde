# Hyperpigmentation-autograde
Columbia University capstone with Unilever. The detailed report could be found in the final reprot folder.
## Abstract
The capstone project with Unilever aims to simulate an auto-grader for grading mottled
hyperpigmentation on the facial images. In this report, we include exploratory data analysis on the image
dataset. Also, we discussed data input pipeline with tensor, and data preprocessing procedures, which
include manual train, validation, and test set splits, data augmentation, and normalization. Then we
discussed three deep learning models: **transfer learning with ResNet50**, **transfer learning with images
after binary thresholding**, and **Siamese** neural networks. Our best practice approaches test RMSE as low
as 0.39 and test accuracy as high as 81.25%. In the end, we discussed potential improvements and future
work.

## Introduction
This capstone project is a collaboration between Columbia University Data Science Institute and Unilever. The
objective of this project is to build a model by leveraging deep learning and image analysis to obtain automatic
grading on mottled hyperpigmentation conditions from facial images taken in clinical studies. In such studies,
subjects’ faces are treated, where different treatments may have different average effects as seen by an expert
grader.

The need for automating the judgments arises because of
(1) Limited availability of expert graders
(2) Difference between graders
(3) Some subjectivity and variability of the experts

Previous attempts by Unilever to build multivariate models on image analysis parameters were not successful.
Similarly, deep learning with VGG16 transfer learning did not yield the desired outcome.
Our goal is that other data science methods, such as a knowledge-based approach or combinations of techniques
may have the potential to result in the desired outcome. With a given dataset of 2403 valid image-grade pairs, we
used three different models in our project to compare which model yields the desired outcome.

![progress](https://user-images.githubusercontent.com/23581048/105645050-38a65780-5e67-11eb-871f-2008b4244f73.png)

## Exploratory Data Analysis and Data Preprocessing
**Dataset**
We have 3 different types of data, images (.jpg), grades (.xlsx), and grading criteria (.docx).

**Input Pipeline with Tensor**
- For the grades, after cleaning the outlier and the missing data,  we grouping the extreme values to deal with the imbalanced data.

- For the images, we build the tensor pipeline to do the split, data augmentation and normalization. 

We chose to treat images as tensors and applied the technique of caching and prefetching to maximize efficiency.
Using caching, the transformation before the cache operation will be executed once at the first epoch. Then we
could reuse the data cached in the following epochs. With prefetching, the input pipeline will pre-read the data for
the next step while the current step is running. With tensors and these two methods, the time spent on each training
epoch is reduced from 80 seconds to 15 seconds when training the baseline model, which is around 70% faster than
the generator method.
 - Train, Validation, and Test Set Splits
 
For the cross-validation steps in the future model evaluating part, we prepared the train, validation, and test set
splits manually. The reason is that because of the nature of caching and prefetching, data would be erased from
local memory if it exceeds the capacity, and not all images are loaded at the same time.
 - Data Augmentation
 
Since a great amount of data is required in training parameters of the transfer learning model, we chose to randomly
flip images horizontally to amplify the dataset and increase the amount of relevant data. The strategy of horizontal
random flip can also help to avoid overfitting.
 - Normalization
 
After data augmentation, we also chose to normalize the augmented dataset prior to training all of the following
models. By doing so, we converted the image pixels to range from -1 to 1, as the original pixels range from 0 to
255. The normalization enables neural networks to work with data easier by improving computation efficiency.

## Modeling
### Baseline model 
 After the preprocessing steps, we chose to implement Transfer Learning. As mentioned in the previous section, our
region of interest is the mottled hyperpigmentation, which is the detailed information on faces. Among the set of
Transfer Learning models, ResNet50 performs better on detecting image details instead of image shapes or edges.
Thus, we used ResNet50 as our baseline model.
![WechatIMG230](https://user-images.githubusercontent.com/23581048/105645094-7a370280-5e67-11eb-83b3-2f206457c409.png)

We trained 50 epochs with batch size 32 and shuffle buffer size 1024. Figure 3.2 and Figure 3.3 show that we
achieved 99.94% in training accuracy and around 52% validation accuracy. From the test data, we achieved 50.42% accuracy.

![WechatIMG231](https://user-images.githubusercontent.com/23581048/105645095-7a370280-5e67-11eb-9eba-34e2cb64c654.png)

According to the confusion matrix, shown in Figure 3.4, we found that most of the misclassifications are near the
diagonal. That is, whenever misclassification occurs, our predicted label is not that far away from the true label. For
example, images in group 3 (represents actual grade 4) are mostly misclassified as group 4 (represents actual grade
4.5).

 ![WechatIMG232](https://user-images.githubusercontent.com/23581048/105645096-7a370280-5e67-11eb-8ee3-9113fb3b79f1.png)

### Method 1: Image Subtraction and Binary Threshold
Model Construction and Results
Instead of using all three channels of the original images, we came up with a new idea of extracting the
hyperpigmentation area of the face images. Our objective is to extract regions of interest (ROI), that is, the mottled
hyperpigmentation area on the face. This enabled the convolutional neural network (CNN) to learn more about the
detailed region of interest instead of face contour, fine lines, and wrinkles.
The idea is inspired by Automatic Acne Detection for Medical Treatment. Our proposed procedure is as follows:

1. Image Subtraction
 - Convert RGB image to Grayscale color space to simplify the calculation.
 - Convert RGB image to HSV color space and extract brightness value (V) from HSV image
 - Subtract grayscale value out of normalized brightness value to reveal a region of interest.
 - Multiply the binary image with the brightness layer of the HSV to get the brightness intensity
 binary threshold image
 
![WechatIMG233](https://user-images.githubusercontent.com/23581048/105645098-7a370280-5e67-11eb-8c3f-ad1ae0f95c5a.png)

2. Binary Thresholding to obtain a binary image

  - Set the threshold value as 0.165 in this case for the best visual result. Any region above 0.165 is
  considered to be the region of interest, and any region below 0.165 is ignored.
  accuracy.
 
  - We took a deeper look at another two images with grade 3 and grade 6 respectively, using binary
  threshold value 0.165. The image of grade 6 in the original RGB color space shows a larger and
  darker area of mottled hyperpigmentation. The images after subtraction and binary thresholding
  show a similar pattern. The binary image with grade 6 has more white pixels than the one with
  
![WechatIMG234](https://user-images.githubusercontent.com/23581048/105645099-7a370280-5e67-11eb-89d0-165f5beb64f7.png)

The model structure and parameter settings are similar to our
baseline model. The only difference is that we trained 100 epochs here. The results are as follows.
grade 3. Also, its white spots are more widely distributed.

### Method 2: Siamese Network
In order to solve the data lacking problem, we tried a new network called Siamese Network. A Siamese Neural
Network is a class of neural network architectures that contains two or more identical subnetworks. ‘ Identical ’ in
this case is defined as having the same configuration with the same parameters and weights. Parameter updating is
mirrored across both sub-networks. Siamese Neural Networks can find the similarity of the inputs by comparing its
feature vectors. So even with a few images, we could get better predictions. For every single image, the network
compares it with all other images from data sources, and learns their similarity. For example, if we have 100
images, for every single image, the network generates 99 other comparisons, which incredibly increases the training
size.

![WechatIMG235](https://user-images.githubusercontent.com/23581048/105645100-7acf9900-5e67-11eb-94d4-dce01579e023.png)

#### Data preparation
Due to the structure of the Siamese network, which needs two input images at one time, we needed to build
appropriate datasets to fit it. In order to do so, we created two identical tensor datasets from a training set with
caching and prefetching to maximize efficiency. We applied the shuffle to both of them and used a detect function
to check whether their labels are identical or not. Then we fed both of them into the Siamese network.

#### Siamese network structure
The network itself, defined in the Net class, is a Siamese convolutional neural network consisting of 2 identical
subnetworks. Here we used our baseline model using transfer learning with ResNet50 . After passing through the
convolutional layers, we let the network build a one-dimensional descriptor of each input by flattening the features
and passing them through a linear layer with 224 output features. Note that the layers in the two subnetworks share
the same weights. This allows the network to learn meaningful descriptors for each input and makes the output
symmetrical (the ordering of the input should be irrelevant to our goal).

The crucial step of the whole procedure is the next one: we calculated the Euclidean distance of the feature vectors.
Usually, the Siamese network performs binary classification at the output, classifying if the inputs are of the same
class or not. Hereby, different loss functions may be used during training. One of the most popular loss functions is
the binary cross-entropy loss. This loss can be calculated as


where L is the loss function, y is the class label (0 or 1) and p is the prediction.
In principle, to train the network, we used binary cross-entropy loss. Therefore, we attached one more linear layer


#### Basic Algorithm
  1. We utilized two images ( Image1 and Image2 ). Both of the images are fed to a single Convolutional
  Neural Network (CNN).
  2. The last layer of the CNN produces a fixed-size vector. Since two images are fed, we got two
  embeddings ( h1 and h2 ).
  3. The Euclidean distance between the vectors was calculated. We used the Lambda layer to calculate
  the absolute distance between the vectors.
  4. The values then passed through a sigmoid function and a similarity score was produced. Since our
  output is binary in nature, we used the tf.keras.losses.binary_crossentropy() function.
  with 2 output features (equal number, different number) to the network to obtain the logits.
  
  ![WechatIMG236](https://user-images.githubusercontent.com/23581048/105645101-7acf9900-5e67-11eb-91e3-2bb54dc76892.png)
  

Since the output of the Siamese network is a similarity score of two images we fed in, another function is required
to do the prediction of the classification of the input image X. Here is how we constructed the classification
problem.

![WechatIMG237](https://user-images.githubusercontent.com/23581048/105645102-7acf9900-5e67-11eb-87a5-e235355c34f1.png)

#### To perform actual predictions on images
  1. Group images from the data set based on their class
  2. Calculate the similarities between the sample image and images for each class
  3. Find the class which has the highest average similarity with the sample image
  4. The class with the highest similarity will be the class prediction assigned to the sample image.
  
  ![WechatIMG238](https://user-images.githubusercontent.com/23581048/105645103-7acf9900-5e67-11eb-98b1-225c942ae374.png)
  
  ![WechatIMG239](https://user-images.githubusercontent.com/23581048/105645104-7acf9900-5e67-11eb-82d2-767665bb3049.png)

To train the model we used Adam as our optimizer and Sparse Categorical Cross-Entropy as our loss function.
Different from the baseline model, we used 1e-8 as our learning rate instead of the cyclical learning rate, because
we rarely faced the local optimal problem when we trained the Siamese network. We used both the Accuracy and
RMSE as the evaluation metrics.
After running 100 epochs on the Siamese network with batch size 64, we achieved 86.4% training accuracy and
84.1% validation accuracy. Compared to the base model and the binary threshold method, the accuracy gains a
great improvement.

![WechatIMG240](https://user-images.githubusercontent.com/23581048/105645105-7b682f80-5e67-11eb-9cd9-8176f03d9857.png)

## Conclusion
In conclusion, we developed an automatic grader for grading mottled hyperpigmentation from facial images taken
in clinical studies. To achieve this goal, we attempted several deep learning models. As a result, we employed three
models:
  1. Baseline Model: a transfer learning model with ResNet50
  2. Binary thresholding as image preprocessing in the baseline model
  3. Siamese Neural Network model with the baseline model as subnetworks
We chose the Siamese Neural Network model to be our final model because the Siamese Neural Network model
outperformed the other two models and achieved test accuracy as high as 81.25% and RMSE (root mean squared
error) as low as 0.39.
In the future, Unilever could follow the proposed future work in the next section.

## Future Work
### Collect More Data With Extreme Grades
Because of the lack of data, we grouped the data with extremely low and high grades into two groups in order to
make the data more uniformly distributed. However, these methods result in lack of precision when predicting cases
of extreme grades. If it is applicable, we would recommend Unilever to collect more data in grades lower than 3,
and grades greater than 5. Hence, the model could have precise predictions for all grades.
### Improve the Baseline Model with Other Transfer Learning Model
We used ResNet50 as our transfer learning model. Throughout our research, we found that other pre-trained models
such as ResNet152 ImageNet Caffe may have better performance. In the research paper Assessing the Severity of
Acne via Cell Phone Selfie Image Using A Deep Learning Model by Hang Zhang, they used ResNet152 ImageNet
Caffe and achieved Root Mean Square Error of 0.482 which is better than the target performance bar 0.517 which
was set by Nestle Skin health and Microsoft as the highest Root Mean Square Error among 11 dermatologists. We
would recommend Unilever to try to train this model and compare the result with ResNet50 .
### Crop Facial Image to Extract Regions of Interest More Precisely
We would recommend Unilever to try to extract skin patches from facial images. From the forehead, both cheeks
and chin of each face image by applying the pre-trained Shape Predictor 68 Face Landmarks (landmark model)
published by Akshayubhat on github.com. The next step is to calculate the weighted average score of the whole
face.
Figure 5.1
