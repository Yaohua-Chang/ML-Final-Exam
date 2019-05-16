## Chapter 1 Introduction
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


5. Which of the following reason is Python the most popular programming languages in data science:
    * a. Good performance for computation-intensive tasks
    * b. A large number of useful add-on libraries.
    * c. Numeric computation libraries built on C code allow for efficient and fast scripts.
    * d. Python has great developers and open-source community.

Answer: (a)

## Chapter 2 Simple ML for Classification, Perceptron, Adaline
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

## Chapter 3 Tour of ML Classifiers, Logistic Regression, SVM


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


## Chapter 4 Decision Tree, KNN, Random Forest
1. What is Not the advantage of decision trees?
    * a. Simple to understand and to interpret
    * b. Effective in high dimensional spaces
    * c. Requires little data preparation
    * d. Performs well even if its assumptions are somewhat violated by the true model 
        from which the data were generated

Answer: (b)

2. A decision tree can be used to build models for 
    * a. Classification problems
    * b. Regression problems
    * c. None of the aboven
    * d. Both of the above
Answer: (d)

3.  To reduce underfitting of a Random Forest model, which of the following method can be used?
    * a. Increase minimum sample leaf value
    * b. Increase depth of trees
    * c. Increase the value of minimum samples to split
    * d. None of these

Answer: (b)


## Chapter 5 Types of Data, One Hot encoding, Feature Selection
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

## Chapter 6 Dimension Reduction
1. Which of the following is not fundamental dimensionality reduction technique for feature extraction?
    * a. ICA
    * b. standard PCA
    * c. LDA
    * d. kernel PCA  

Answer: (a)

2. Which of the following algorithms cannot be used for reducing the dimensionality of data?
    * a. t-SNE
    * b. PCA
    * c. LDA False
    * d. None of these

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

## Chapter 7 Hyper-parameters, Pipelines, Training/evaluation, overfitting
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



## Chapter 8 Ensemble Methods, Bagging, Boosting
1. Which of the following description is wrong for ensemble methods? 

    * a. Ensemble learning increases the computational complexity compared to individual classifiers.
    * b. The ensemble can be built from different classification algorithms
    * c. the ensemble can be built from same base classification algorithm
    * d. Ensemble methods always can work better than individual classifiers alone

Answer: (d)

2. Which of the following is woring for Bagging? 

    * a. Bagging algorithm can be an effective approach to reduce the variance of a model
    * b. Bagging is ineffective in reducing model bias
    * c. Bagging is also known as bootstrap aggregating
    * d. Bagging is an ensemble learning technique that is closely related to the MajorityVoteClassifier.

Answer: (b)

3. Which of the following algorithms is scale-invariant so that we don't need to apply StandardScaler into it?
    * a. Logistic Regression
    * b. k-nearest neighbors
    * c. Decision Trees
    * d. k-means

Answer: (c)

4. Which of the following is woring for Boosting? 

    * a. Boosting consists of very simple base classifiers。
    * b. Boosting let the weak learners subsequently learn from misclassified training samples to improve the performance of the ensemble.
    * c. Boosting can lead to a decrease in bias as well as variance compared to bagging models.  
    * d. A typical example of a strong learner is a decision tree stump. 

Answer: (d)

5. Which of the following is not common way to address overfitting? 

    * a. Increase the number of parameters of the model
    * b. Collect more training data
    * c. Reduce the complexity of the model
    * d. Increase the regularization parameter

Answer: (a)


## Chapter 9 Regression and Clustering
1. Which of the following description is wrong for ensemble methods? 

    * a. Cluster is a category of supervised learning techniques.
    * b. Cluster allows us to discover hidden structures in data where we do not know the right answer upfront. 
    * c. The goal of clustering is to find a natural grouping in data so that items in the same cluster are more similar to each other than to those from different clusters.
    * d. Clustering is a technique that allows us to find groups of similar objects, objects that are more related to each other than to objects in other groups.


Answer: (a)

2. Which of the following is not example of business-oriented applications of clustering? 

    * a. Grouping of documents, music, and movies by different topics.
    * b. Finding customers that share similar interests based on common purchase behaviors as a basis for recommendation engines.
    * c. Recommanding custom difference information according to their interest.
    * d. Using in outlier detection applications such as detection of credit card fraud.

Answer: (c)

3. Which type of clustering does k-means belong to?
    * a. Hierarchical clustering 
    * b. Distance-based clustering
    * c. Prototype-based clustering
    * d. Density-based clustering

Answer: (c)

4. Which of the following description is woring for k-means? 

    * a. k-means is not good at identifying clusters with a spherical shape.
    * b. we have to specify the number of clusters for k-means.
    * c. A problem with k-means is that one or more clusters can be empty. 
    * d. Within-cluster Sum of Squared Errors (SSE) is sometimes also called cluster inertia. 

Answer: (a)

5. Which of the following description is woring for k-means++? 

    * a. k-means++ can sometimes result in bad clusterings or slow convergence if the initial centroids are chosen poorly. 
    * b. k-means++ doesn't address those assumptions and drawbacks of k-means
    * c. k-means++ can greatly improve the clustering results through more clever seeding of the initial cluster centers.
    * d. k-means++ place the initial centroids far away from each other, which leads to better and more consistent results than the classic k-means.

Answer: (a)

## Chapter 10 Clutering, Data Mining, Latent Model, Collab Filtering

## Chapter 11 Multi-Layer Neural Network
1. Which of the following description is wrong for Multilayer Perceptron (MLP)? 

    * a. We can think of the neurons in the MLP as logistic regression units that return values in the continuous range between 0 and 1.
    * b. MLP is a typical example of a feedforward artificial neural network. 
    * c. MLP depicted in the preceding figure has one input layer, one or more hidden layer, and one output layer.
    * d. To be able to solve complex problems such as image classification, we need non-linear activation functions in our MLP model, for example, the sigmoid (logistic) activation function.

Answer: (c)

2. Which of the following is not example of applications of  deep neural networks(DNNs)? 

    * a. Google's image search
    * b. Google Translate
    * c. A mobile application that can detect skin cancer with an accuracy similar to professionally trained dermatologists
    * d. Outlier detection applications such as detection of credit card fraud.

Answer: (d)


3. Which of the following description is woring for multilayer neural networks? 

    * a. Adaline is a typical example of multilayer neural networks.
    * b. Multilayer neural networks are much harder to train than simpler algorithms such as logistic regression, or support vector machines.
    * c. If a neural network has more than one hidden layer, we also call it a deep artificial neural network. 
    * d. MLP is a typical example of a multilayer neural network.

Answer: (a)


## Chapter 12 CNN, RNN
