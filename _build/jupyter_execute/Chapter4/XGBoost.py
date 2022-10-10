#!/usr/bin/env python
# coding: utf-8

# # Gradient Boost
# 
# ## Basic Gradient Boosting
# 
# `AdaBoost` as an algorithm to emphasize on the wrong answers by increasing the weights of wrong answers. Gradient Boost is to directly train the model over the differnce between the predicted answers and the labels. When the base classifier is a Decision Tree, the resulted model is called *Gradient Boosted Decision Trees*, or `GBDT` for short.
# 
# The basic idea of `GBDT` is to do regression, that the expected outcomes are continuous numbers. However when you set the class names to be numbers, and intrept the outcome of the model as the probability to be belong to the class, `GBDT` can also be used in classification problems. 
# 
# ## Desciption of the algorithm
# 

# 
# ## XGBoost
# https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
# 
# XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.
# 
# XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
# 
# In this post you will discover XGBoost and get a gentle introduction to what is, where it came from and how you can learn more.
# 
# After reading this post you will know:
# 
# - What XGBoost is and the goals of the project.
# - Why XGBoost must be a part of your machine learning toolkit.
# - Where you can learn more to start using XGBoost on your next machine learning project.
# 
# 
# Let's start by briefly reviewing ensemble learning. Like the name suggests, ensemble learning involves building a strong model by using a collection (or "ensemble") of "weaker" models.  Gradient boosting falls under the category of boosting methods, which iteratively learn from each of the weak learners to build a strong model. It can optimize:
# 
# 
