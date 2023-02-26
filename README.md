# ML Project - PSSVM

Your Tasks:
1. Building a SVM Classifier
    - Pre-processing the data:
    i. Randomly pick 80% of the data as a training set and the rest as a test set.
    ii. Normalize each feature of the dataset to have zero mean and unit variance. Note that
    while normalizing the features, their mean and variance should be computed over the train
    split only. Once, the mean and variance is computed using only the train split, you
    normalize the test split using the mean and variance computed over the train split.
    - Training the model:
      i. Note that training requires solving the dual optimization problem. To solve the dual
      optimization problem you can use any python packages like: CVXOPT or Scipy.optimize.minimize
      ii. Implement the following three kernels : linear, quadratic, radial basis function
    - Making predictions: Write a function that takes new datapoint as input and predicts the class
    - Evaluation: Finally, you should generate results on the given data and compare its results with
    the sklearn module (sklearn.svm)
  
2. Hyper-parameter Tuning: 
 In Support Vector Machines, the Gamma and C hyper-parameters control the model's
 flexibility and generalization performance.
    - Choose 5 values of Gamma and 5 values of C. (Justify your choice in report). Tune the
    hyper-parameters by doing a grid search on all combinations of (kernel, gamma, C).
    - Report test set accuracy for all the 3x5x5=75 combinations in a tabular form.

3. Visualization
    - Consider the model (with best hyper-parameters) and plot the the decision boundary and the
    support vectors, on both train & test set. (You may use suitable python packages for this task)

4. Report
    - Prepare a concise report or manual, spanning 2 to 3 pages, that details the results of your
    work, along with your observations and explanations. Be sure to include a clear and thorough
    overview of the methods used, the results obtained, and your interpretation of the findings.

## Team Members
- Jaisaikrishnan
- Madhusudan Agarwal
- Chillara Aditya Saikumar
