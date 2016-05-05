Kaggle: Yelp Restaurant Photo Classification
============================

Each directory in this repository includes a submission to the Kaggle competition [Yelp Restaurant Photo Classification](https://www.kaggle.com/c/yelp-restaurant-photo-classification).

## Three Submissions:

- [/CNN_Submission1](https://github.com/ncchen55414/Kaggle-Yelp/tree/master/CNN_Submission1):
This starter code uses CaffeNet to extract image features, takes the mean of image features to represent restaurants, and uses SVM to classify restaurants. Private LB F1-score = 0.76094. [(More Discussion)](https://www.kaggle.com/c/yelp-restaurant-photo-classification/forums/t/19206/deep-learning-starter-code)

- [/HOG_Submission1](https://github.com/ncchen55414/Kaggle-Yelp/tree/master/HOG_Submission1):
[Benedict](https://github.com/thebenedict)'s submission; similar to CNN_Submission but uses histogram of oriented gradients (HOG) features. 

- [/BagDistance_Submission1](https://github.com/ncchen55414/Kaggle-Yelp/tree/master/BagDistance_Submission1): This computes the Chamfer distances between restaurants and uses SVM classifiers. Private LB F1-score = 0.82219
