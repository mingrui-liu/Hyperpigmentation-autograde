# Hyperpigmentation-autograde
Columbia University capstone with Unilever 
## Abstract
The capstone project with Unilever aims to simulate an auto-grader for grading mottled
hyperpigmentation on the facial images. In this report, we include exploratory data analysis on the image
dataset. Also, we discussed data input pipeline with tensor, and data preprocessing procedures, which
include manual train, validation, and test set splits, data augmentation, and normalization. Then we
discussed three deep learning models: transfer learning with ResNet50, transfer learning with images
after binary thresholding, and Siamese neural networks. Our best practice approaches test RMSE as low
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

