{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gradient Boost\n",
    "\n",
    "## Basic Gradient Boosting\n",
    "\n",
    "`AdaBoost` as an algorithm to emphasize on the wrong answers by increasing the weights of wrong answers. Gradient Boost is to directly train the model over the differnce between the predicted answers and the labels. When the base classifier is a Decision Tree, the resulted model is called *Gradient Boosted Decision Trees*, or `GBDT` for short.\n",
    "\n",
    "The basic idea of `GBDT` is to do regression, that the expected outcomes are continuous numbers. However when you set the class names to be numbers, and intrept the outcome of the model as the probability to be belong to the class, `GBDT` can also be used in classification problems. \n",
    "\n",
    "## Desciption of the algorithm\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "## XGBoost\n",
    "https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/\n",
    "\n",
    "XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.\n",
    "\n",
    "XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.\n",
    "\n",
    "In this post you will discover XGBoost and get a gentle introduction to what is, where it came from and how you can learn more.\n",
    "\n",
    "After reading this post you will know:\n",
    "\n",
    "- What XGBoost is and the goals of the project.\n",
    "- Why XGBoost must be a part of your machine learning toolkit.\n",
    "- Where you can learn more to start using XGBoost on your next machine learning project.\n",
    "\n",
    "\n",
    "Let's start by briefly reviewing ensemble learning. Like the name suggests, ensemble learning involves building a strong model by using a collection (or \"ensemble\") of \"weaker\" models.  Gradient boosting falls under the category of boosting methods, which iteratively learn from each of the weak learners to build a strong model. It can optimize:\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.9.12 ('ml22')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.12"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "4eae2d79809986d0872e4e364459f0c9575ffff27a18380d5ee1c7bc910cc873"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
