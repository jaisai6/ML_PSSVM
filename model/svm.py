# Group: 10
# Rollno: 19EE10050, 19EC10041, 22CS60R18
# Project no: 
# Project Title: 


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
from time import process_time


class Preprocessor:
    '''Preprocessor'''

    def __init__(self):
        self.means = {}
        self.stds = {}

    def get_folds(self, p_csv, prediction_class):
        '''
            input: processed csv, prediction class
            output: 5 folds
            
            Each fold contains (X_train, y_train, X_test, y_test)
            X_train, X_test are nd-arrays
            y_train, y_test are 1d-arrays

            Usage: see task_1()
        '''
        folds = []
        for train_index, test_index in self.__perform_5_fold(p_csv):
            train_set = p_csv.iloc[train_index]
            test_set = p_csv.iloc[test_index]

            # seperate prediction class from other classes
            X_train, y_train = self.separate_prediction_class(train_set, prediction_class)
            X_test, y_test = self.separate_prediction_class(test_set, prediction_class)

            X_train = X_train.to_numpy()
            y_train = y_train[prediction_class].to_numpy()
            X_test = X_test.to_numpy()
            y_test = y_test[prediction_class].to_numpy()

            folds.append((X_train, y_train, X_test, y_test))
        
        return folds

    def __perform_5_fold(self, dataset):
        '''
            input: train set
            output: generator object which generates one fold indices in one iteration.
        '''

        k_fold = KFold(n_splits=5, shuffle=True, random_state=6)
        for train_index, test_index in k_fold.split(dataset):
            yield train_index, test_index

    def normalize_columns(self, df, prediction_class, test=False):
        '''
            Standardization (Z-score normalization)
            x = (x - mean) / sd
        '''
        for column in df:
            if column == prediction_class:
                ind = (df[column] == 0)
                df[column][ind] = -1
                continue
                
            mean = None
            std = None
            if not test:
                self.means[column] = df[column].mean()
                self.stds[column] = df[column].std()
            mean = self.means[column]
            std = self.stds[column]
            df[column] = (df[column] - mean) / std

    def drop_outliers(self, X_csv, Y_csv):
        limit = {}
        drop_rows = []
        n_columns = len(X_csv.columns)
        for column in X_csv.columns:
            Mean = X_csv[column].mean()
            Std = X_csv[column].std()
            limit[column] = Mean + 3 * Std

        print('limits done')
        for i, row in X_csv.iterrows():
            n_col_outliers = 0
            for column in X_csv.columns:
                if (row[column] > limit[column]):
                    n_col_outliers += 1
            if (2 * n_col_outliers > n_columns):
                drop_rows.append(i)
        print('dropping rows...')
        return X_csv.drop(drop_rows), Y_csv.drop(drop_rows)

    def separate_prediction_class(self, dataset, prediction_class):
        '''
            input: prediction class
            output: X df, y df
            Seperates prediction class from other classes
        '''
        Y = dataset[[prediction_class]]
        X = dataset.drop([prediction_class], axis=1)
        return X, Y

    def process(self, csv_file, prediction_class):
        '''
            input: raw csv file
            output: prcessed csv (dataframe)

            1. Normalizes features [0,1]

            Usage:
                p = Preprocessor()
                p_csv = p.process('hospital.csv', 'Stay')
        '''
        # read input
        df = pandas.read_csv(csv_file)

        # 14319 rows required 1.6GB of memory to store Kernel Matrix
        df = df.iloc[:6000, :]

        # split dataset into train and test
        train, test = train_test_split(df, test_size=0.2, random_state=6)

        # normalize columns
        self.normalize_columns(train, prediction_class)
        self.normalize_columns(test, prediction_class, test=True)

        # separate prediction class
        X_train, Y_train = Preprocessor().separate_prediction_class(train, 'Class')
        X_test, Y_test = Preprocessor().separate_prediction_class(test, 'Class')

        return X_train, Y_train.iloc[:, 0], X_test, Y_test.iloc[:, 0]
    
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def visualize_boundary():
    X_train, y_train, X_test, y_test = Preprocessor().process("dataset/pulsar_star_dataset.csv", "Class")
    model = svm.SVC(kernel='linear')
    clf = model.fit(X_train.iloc[:, :2], y_train)
    y_pred = model.predict(X_test.iloc[:, :2])
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    title = "SVC with linear kernel"

    X0, X1 = X_train.iloc[:, 0], X_train.iloc[:, 1]
    y = y_train

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots()

    # setup grid
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel(X_train.columns[0])
    ax.set_xlabel(X_train.columns[1])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()


def sklearn_svm():
    X_train, y_train, X_test, y_test = Preprocessor().process("dataset/pulsar_star_dataset.csv", "Class")
    model = svm.SVC(kernel='rbf')
    clf = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


def linear_kernel(x1, x2, gamma=0.1):
    return np.dot(x1, x2)


def quadratic_kernel(x1, x2, gamma=0.1):
    return (np.dot(x1, x2)+1) ** 2

def rbf_kernel(x1, x2, gamma=0.1):
    """Computes the RBF kernel between two matrices of feature vectors"""

    # Compute the pairwise squared Euclidean distances
    dist = np.dot((x1 - x2).T, (x1 - x2))

    # Compute the kernel matrix
    K = np.exp(-gamma * dist)

    return K


############################################################################
from scipy.optimize import minimize

# returns negation of actual equation so that it can be minimized
def objective(alphas, n, x, y, K):
    alphas_sum = np.sum(alphas)
    second_term = 0
    for i in range(n):
        for j in range(n):
            second_term += alphas[i] * alphas[j] * y[i] * y[j] * K[i][j]
    ret = alphas_sum - 0.5 * second_term    
    return -(ret)

def constraint(alphas, n, y):
    total = 0
    for i in range(n):
        total += alphas[i] * y[i]
    return total
############################################################################


from cvxopt import matrix, solvers

def solve_dual_opt_probelm(x, y, kernel_function, c=1000, gamma=0.1):
    m, n = x.shape
    
    # precompute kernel_fuction values
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            K[i][j] = kernel_function(x[i], x[j], gamma)
            # symmetric kernel function
            K[j][i] = K[i][j]
    print('done computing kernel matrix.')

    y = y.astype(float)
    # z = np.atleast_2d(y)
    # coeff = np.matmul(z.T, z)
    coeff = np.outer(y, y)
    P = matrix(coeff * K)
    q = matrix(np.ones((m, 1)) * -1)
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    # G = matrix(np.eye(m) * -1)
    # h = matrix(np.zeros(m))
    G = matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * c)))

    opts = {'maxiters' : 10}
    solution = solvers.qp(P, q, G, h, A, b, options=opts)
    return solution
    
    # minimize
    # sol = minimize(objective, alphas, args=(n, x, y, K), method='SLSQP', bounds=bounds, constraints=constraints)
    # return sol

def calculate_weights(alphas, x, y):
    n = len(x)
    d = len(x[0])
    
    w = np.zeros(d)
    for i in range(d):
        for j in range(n):
            w[i] += alphas[j] * y[j] * x[j][i]
    return w

def calculate_bias(w, p, o):
    return (o - np.dot(w, p))

def predict(w, b, X_test):
    y_pred = []
    for x in X_test:
        if np.dot(w, x) + b > 0:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred

def task_1():
    X_train, y_train, X_test, y_test = Preprocessor().process("dataset/pulsar_star_dataset.csv", "Class")
    
    t1 = process_time()

    # Example-1:
    # x = ((3, 1), (3, -1), (6, 1), (6, -1), (1, 0), (0, 1), (0, -1), (-1, 0))
    # y = (1 ,1 , 1, 1, -1, -1, -1, -1)
    # x, y = np.array(x), np.array(y)
    
    # train
    x, y = X_train.to_numpy(), y_train.to_numpy()
    print('dataset size: {}, dimension: {}'.format(len(x), len(x[0])))

    # sol = solve_dual_opt_probelm(x, y, linear_kernel, c=1)
    sol = solve_dual_opt_probelm(x, y, linear_kernel, c=1, gamma=0.125)
    alphas = np.array(sol['x'])
    ind = (alphas > 1e-6).flatten()
    print('support vectors:')
    print(x[ind])
    w = calculate_weights(alphas, x, y) # Approximation is not done while calculating weights
    b = calculate_bias(w, x[ind][0], y[ind][0]) # Approximation is done while calculating bias
    t2 = process_time()

    print(w, b)

    y_pred = predict(w, b, X_test.to_numpy())
    print(accuracy_score(y_test.to_list(), y_pred))

    print("Training Time: ", t2 - t1)

if __name__ == "__main__":

    # sklearn_svm()
    # visualize_boundary()
    task_1()
