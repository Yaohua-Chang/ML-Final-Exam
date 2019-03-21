## Chapter 1
1. Which of the following is not types of machine learning:
    * a. supervised learning
    * b. unsupervised learning
    * c. reinforcement learning
    * d. deep learning

Answer: (d)


2. Which of the following does not belong to a typical workflow:
    * a. Preprocessing
    * b. Learning
    * c. EDA
    * d. Evaluation

Answer: (c)


3. Which of the following can not be used in unsupervised learning:
    * a. Classification
    * b. Regression
    * c. Dimension reduction 
    * d. Clustering

Answer: (b)


4. Which of the following can be used for predicting continuous outcomes:
    * a. Classification
    * b. Regression
    * c. Dimension reduction 
    * d. Clustering

Answer: (b)


5. Which of the following reason is Not Python the most popular programming languages in data science:
    * a. Good performance for computation-intensive tasks
    * b. A large number of useful add-on libraries.
    * c. Numeric computation libraries built on C code allow for efficient and fast scripts.
    * d. Python has great developers and open-source community.

Answer: (a)

## Chapter 2
1. What is the main difference between Adaline and a Perceptron?
    * a. the weights are updated based on a linear activation function rather than a unit step function
    * b. the weights are updated based on a unit step function rather than linear activation function 
    * c. implement different functions
    * d. None of the below

Answer: (a)


2. What is the main advantage of  continuous linear activation function compared to the unit step function?
    * a. the cost function becomes differentiable
    * b. the cost function is that it is convex
    * c. Both of a and b
    * d. None of the below

Answer: (c)


3. Which technique does gradient descent benefit from?
    * a. feature selection
    * b. feature extraction
    * c. feature reduction
    * d. feature scaling

Answer: (d)


4. Which of the following is not a type of gradient descent?
    * a. stochastic gradient descent 
    * b. batch gradient descent
    * c. mini-batch learning
    * d. online learning

Answer: (d)


5. Which of the following is Not the advantage of stochastic gradient descent?
    * a. It can escape shallow local minima more readily if we are working with nonlinear cost functions.
    * b. It is useful for online learning.
    * c. It can be considered as an approximation of gradient descent.
    * d. It typically reaches convergence much faster.

Answer: (c)

## Chapter 3

1. The effectiveness of an SVM depends upon:
    * a. Selection of Kernel
    * b. Kernel Parameters
    * c. Soft Margin Parameter C
    * d. All of the above

Answer: (d)

2.  What do you mean by a hard margin?
    * a. The SVM allows high amount of error in classification
    * b. The SVM allows very low error in classification
    * c. Widen the margin and allows for violation
    * d. None of the above

Answer: (b)

3. What is Not the advantage of decision trees?
    * a. Simple to understand and to interpret
    * b. Effective in high dimensional spaces
    * c. Requires little data preparation
    * d. Performs well even if its assumptions are somewhat violated by the true model from which the data were generated

Answer: (b)

4. A decision tree can be used to build models for 
    * a. Classification problems
    * b. Regression problems
    * c. None of the aboven
    * d. Both of the above
Answer: (d)

5.  To reduce underfitting of a Random Forest model, which of the following method can be used?
    * a. Increase minimum sample leaf value
    * b. Increase depth of trees
    * c. Increase the value of minimum samples to split
    * d. None of these

Answer: (b)


## Chapter 4
1. Which strategy parameter is best choice for imputing categorical feature values when we deal with NaN value?
    * a. mean
    * b. median
    * c. mass
    * d. most_frequent  

Answer: (d)

2. Which of the following is not part of data processing?
    * a. Imputing missing values
    * b. Feature scaling
    * c. Partitioning a dataset into separate training and test sets
    * d. EDA

Answer: (d)

3. Which of the following is not common way to select features?
    * a. L1 regularization
    * b. L2 regularization
    * c. random forest
    * d. PCA

Answer: (d)

4. Which following solution is often not applicable to reduce the generalization error ?
    * a. Collect more training data
    * b. Introduce a penalty for complexity via regularization
    * c. Choose a simpler model with fewer parameters
    * d. Reduce the dimensionality of the data

Answer: (a)

5. Which is categorie of dimensionality reduction techniques?
    * a. feature selection
    * b. feature extraction
    * c. Both of the above
    * d. None of the above

Answer: (c)

## Chapter 5
1. Which of the following is not fundamental dimensionality reduction technique for feature extraction?
    * a. ICA
    * b. standard PCA
    * c. LDA
    * d. kernel PCA  

Answer: (a)

2. Which of the following algorithms can be used for reducing the dimensionality of data?
    * a. t-SNE
    * b. PCA
    * c. LDA
    * d. All of the above

Answer: (d)

3. The most popularly used dimensionality reduction algorithm is Principal Component Analysis (PCA). Which of the following is/are true about PCA?
    * a. PCA is an unsupervised method.
    * b. It searches for the directions that data have the largest variance.
    * c. All principal components are orthogonal to each other
    * d. All of the above

Answer: (d)

4. Which of the following statement is correct for t-SNE and PCA?
    * a. t-SNE is linear whereas PCA is non-linear.
    * b. t-SNE is nonlinear whereas PCA is linear
    * c. t-SNE and PCA both are linear
    * d. t-SNE and PCA both are nonlinear

Answer: (b)

5. Which of the following is true about LDA?
    * a. LDA aims to maximize the distance between class and minimize the within class distance.
    * b. LDA aims to minimize both distance between class and distance within class.
    * c. LDA aims to minimize the distance between class and maximize the distance within class.
    * d. LDA aims to maximize both distance between class and distance within class.

Answer: (a)

## Chapter 6
1. What’s not the part of streamlining workflows with pipelines? 

    * a. Scaling dataset
    * b. Dimensionality Reduction
    * c. Spliting dataset into training set and test set
    * d. Learning Algorithm

Answer: (c)

2. What’s part of dataset worked for model selection in holdout method? 

    * a. Training dataset
    * b. Validation dataset
    * c. Test dataset
    * d. Original dataset

Answer: (b)

3. Which of the following options is/are true for K-fold cross-validation?
    * a. Increase in K will result in higher time required to cross validate the result.
    * b. Higher values of K will result in higher confidence on the cross-validation result as compared to lower value of K.
    * c. If K=N, then it is called Leave one out cross validation, where N is the number of observations.
    * d. All of the above  

Answer: (d)

4. Which of the following is true about bias and variance? 

    * a. High bias model means overfitting.
    * b. High variance model means underfitting.
    * c. If our model is too simple and has very few parameters then it may have low bias and low variance. 
    * d. If our model has large number of parameters then it’s going to have high variance and low bias.

Answer: (d)

5. Which of the following is not common way to address overfitting? 

    * a. Increase the number of parameters of the model
    * b. Collect more training data
    * c. Reduce the complexity of the model
    * d. Increase the regularization parameter

Answer: (a)

