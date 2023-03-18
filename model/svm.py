# Group: 10
# Rollno: 19EE10050, 19EC10041, 22CS60R18
# Project Code: PSSVM
# Project Title: Pulsar Star Classification using Support Vector Machines


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from tqdm import tqdm

from cvxopt import matrix, solvers


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
        df = df.iloc[:2000, :]

        # split dataset into train and test
        train, test = train_test_split(df, test_size=0.2, random_state=6)

        # normalize columns
        self.normalize_columns(train, prediction_class)
        self.normalize_columns(test, prediction_class, test=True)

        # separate prediction class
        X_train, Y_train = Preprocessor().separate_prediction_class(train, 'Class')
        X_test, Y_test = Preprocessor().separate_prediction_class(test, 'Class')

        return X_train, Y_train.iloc[:, 0], X_test, Y_test.iloc[:, 0]
    
class SVM:

    def __init__(self, kernel='linear', C=1.0, gamma=0.125):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.alphas = None
        self.weights = None
        self.bias = None
        self.support_vectors = []

    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)

        elif self.kernel == 'quadratic':
            return (np.dot(x1, x2)+1) ** 2 

        else:
            dist = np.dot((x1 - x2).T, (x1 - x2))
            K = np.exp(-self.gamma * dist)
            return K

    def calculate_weights(self, alphas, x, y):
        y = y.reshape(-1,1) # Convert the y into same form as alphas... dim(alphas) : m * 1
        w = ((y*alphas).T)@x
        return w

    def calculate_bias(self, w, x_sv, y_sv):
        return (y_sv - np.dot(x_sv,w))

    def qp_solver(self, x, y):
        m, n = x.shape # m : no. of examples in training data, n : no. of attributes in training data
        K = np.zeros((m, m))
        for i in tqdm(range(m)):
            for j in range(i, m):
                K[i][j] = self.kernel_function(x[i], x[j])
                K[j][i] = K[i][j] # symmetric kernel function

        y = y.astype(float)
        coeff = np.outer(y,y)
        P = matrix(coeff * K)
        q = matrix(np.ones((m, 1)) * -1)
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
        G = matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        opts = {'maxiters' : 10, 'show_progress' : True}
        solution = solvers.qp(P, q, G, h, A, b, options=opts)
        return solution

    def fit(self, X_train, y_train):
        print("--------------------------------Training--------------------------------")
        t1 = process_time()
        x, y = X_train.to_numpy(), y_train.to_numpy()
        print('dataset size: {}, dimension: {}'.format(len(x), len(x[0])))

        sol = self.qp_solver(x, y)
        alphas = np.array(sol['x'])

        sv1 = np.where(alphas <= 1 + 1e-15)
        sv2 = np.where(alphas > 1e-2)
        self.support_vector_indices = np.intersect1d(sv1, sv2)
        sv = (alphas >= 1e-6).flatten()
        x_sv = x[sv]
        y_sv = y[sv]

        weights = self.calculate_weights(alphas, x, y) # Using the entire dataset to calculate weights
        b = self.calculate_bias(weights[0], x_sv, y_sv) # Using only the support vectors to calculate b vector
        bias = np.sum(b)/b.size  # Take average over all the support vectors to get the bias

        for p in x:
            value = np.abs(np.dot(weights, p) + bias)
            if value >= 0.9 and value <= 1 + 1e-15:
                self.support_vectors.append(p)
        self.support_vectors = np.array(self.support_vectors)

        t2 = process_time()

        print('\nResults ')
        print('Weights: ', weights)
        print('Bias: ', bias)
        print("\nTraining Time: ", t2 - t1)
        print("------------------------------------------------------------------------")

        self.weights = weights
        self.bias = bias
        self.alphas = alphas
        return weights, bias

    def predict(self, X_test):
        X_test = X_test if isinstance(X_test, np.ndarray) else X_test.to_numpy()
        y_pred = []
        for x in X_test:
            if np.dot(self.weights, x) + self.bias > 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return np.array(y_pred)

    def score(self, y_test, y_pred):
        print("\nAccuracy: ", accuracy_score(y_test.to_list(), y_pred)) # Accuracy

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, model, xx, yy, **params):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def visualize_boundary(kernel='linear', C=1, gamma=0.125, is_sklearn=True, is_test=False):
    X_train, y_train, X_test, y_test = Preprocessor().process("pulsar_star_dataset.csv", "Class")
    columns = [2, 5]
    X_train = X_train.iloc[:, columns]
    X_test = X_test.iloc[:, columns]

    model = svm.SVC(kernel='poly' if kernel=='quadratic' else kernel, C=C, gamma=gamma, degree=2) if (is_sklearn == True) \
           else SVM(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    title = "SVM with " + kernel + " kernel (Train)" 
    title_t = "SVM with " + kernel + " kernel (Test)" 
    
    X0, X1 = X_train.iloc[:, 0], X_train.iloc[:, 1]
    y = y_train

    X0_t, X1_t = X_test.iloc[:, 0], X_test.iloc[:, 1]
    y_t = y_test   

    # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots()
    fig_t, ax_t = plt.subplots()

    # setup grid
    xx, yy = make_meshgrid(X0, X1)
    # xx_t, yy_t = make_meshgrid(X0_t, X1_t)

    plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    plot_contours(ax_t, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax_t.scatter(X0_t, X1_t, c=y_t, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    
    # support vectors
    support_vector_indices = None
    v = None
    if is_sklearn:
        # obtain the support vectors through the decision function
        decision_function = model.decision_function(X_train)
        A = np.where(np.abs(decision_function) <= 1 + 1e-15)
        B = np.where(np.abs(decision_function) > 0.9)
        support_vector_indices = np.intersect1d(A, B)
        v = X_train.to_numpy()[support_vector_indices]
    else:
        v = model.support_vectors
    
    ax.scatter(v.T[0], v.T[1], color=['white' for i in range(len(v))], s=20, edgecolors='k')
    ax_t.scatter(v.T[0], v.T[1], color=['white' for i in range(len(v))], s=20, edgecolors='k')

    ax.set_ylabel(X_train.columns[0])
    ax.set_xlabel(X_train.columns[1])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()

    ax_t.set_ylabel(X_test.columns[0])
    ax_t.set_xlabel(X_test.columns[1])
    ax_t.set_xticks(())
    ax_t.set_yticks(())
    ax_t.set_title(title_t)
    ax_t.legend()
    plt.show()
     


def sklearn_svm():
    X_train, y_train, X_test, y_test = Preprocessor().process("dataset/pulsar_star_dataset.csv", "Class")
    model = svm.SVC(kernel='rbf')
    clf = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

if __name__ == "__main__":

    # sklearn_svm()
    visualize_boundary(kernel='quadratic', C=0.1, is_sklearn=False)
