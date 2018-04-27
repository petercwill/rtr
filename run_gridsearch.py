# imports and jupyter nb settings

import matplotlib.pylab as plt
import pandas as pd
import os
import numpy as np
from scipy import sparse
import math

# constants
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data/")

ORDERS = os.path.join(DATA_DIR, "orders.csv")
REVIEWS = os.path.join(DATA_DIR, "reviews.csv")
SIZE_MAPPINGS = os.path.join(DATA_DIR, "size_mapping.csv")
STYLE_ATTR = os.path.join(DATA_DIR, "style_attributes.csv")
USER_ATTR = os.path.join(DATA_DIR, "user_attributes.csv")

DATE_FRMT = "%Y-%m-%d"

#Read in data
orders = pd.read_csv(ORDERS)
reviews = pd.read_csv(REVIEWS)
#size_mappings = pd.read_csv(SIZE_MAPPINGS)
style_attr = pd.read_csv(STYLE_ATTR)
user_attr = pd.read_csv(USER_ATTR)

def prepare_data(orders_df, reviews_df, thin_users, thin_styles):
    
    """
    Preprocesses data and returns a dense  matrix factorization to predict empty
    entries in a matrix.

    Arguments
    - orders_df           : pandas dataframe containing order information
    - reviews_df          : pandas dataframe containing review information
    - thin_users (float)  : percentage of unique users to keep
    - thin_users (float)  : percentage of unique styles to keep

    """
        
    unique_users = orders['user_id'].unique()
    thinned_users = np.random.choice(unique_users, int(len(unique_users)*thin_users), replace=False)
    
    unique_styles = orders['style'].unique()
    thinned_styles = np.random.choice(unique_styles, int(len(unique_styles)*thin_styles), replace=False)
    
    _orders_df = orders_df.copy()
    _orders_df.drop_duplicates(inplace=True)
    
    _reviews_df = reviews_df.copy()
    _reviews_df.drop_duplicates(inplace=True)
    
    _orders_df = _orders_df[_orders_df['user_id'].isin(thinned_users)]
    _orders_df = _orders_df[_orders_df['style'].isin(thinned_styles)]
    
    _orders_df.set_index("order_id", inplace=True)
    joined = _reviews_df.join(_orders_df, how='left', on="order_id")
    joined = joined[['user_id','style','didnt_fit']]
    joined['didnt_fit'] = joined['didnt_fit'].astype(int)
    joined.loc[joined['didnt_fit'] == 0, 'didnt_fit'] = 10
    return joined.groupby(['user_id','style'])['didnt_fit'].mean().unstack()

def data2sparse(df, training_prop, testing_prop):
    (x_inds, y_inds) = np.nonzero(~np.isnan(df.values))
    n = len(x_inds)
    inds = np.random.choice(n, int((training_prop+testing_prop)*n), replace=False)
    train_inds = np.random.choice(
        inds, int(training_prop/(training_prop + testing_prop)*len(inds)), replace=False
    ) 
    inds = set(inds)
    train_inds = set(train_inds)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(n):
        if i not in inds:
            continue
        else:
            if i in train_inds:
                train_x.append(x_inds[i])
                train_y.append(y_inds[i])
            else:
                test_x.append(x_inds[i])
                test_y.append(y_inds[i])

    train = [train_x, train_y]
    test = [test_x, test_y]
    train_sparse = sparse.dok_matrix(sparse.coo_matrix((df.values[train], train), shape = df.shape))
    test_sparse = sparse.dok_matrix(sparse.coo_matrix((df.values[test], test), shape = df.shape))
    return (train_sparse, test_sparse)


def make_user_attrib_dict(user_attr, grouped_df):
    
    SIZE_INCR = 1
    HEIGHT_INCR = 1
    WEIGHT_INCR = 2
    BMI_INCR = 1
    
    _df = user_attr.copy()
    _df['bmi'] = (_df['weight_lbs'] / _df['height_in'].pow(2))*703
    _df['size_bin'] = np.floor((_df['standard_size']  - _df['standard_size'].min()) / SIZE_INCR)
    _df['height_bin'] = np.floor((_df['height_in']  - _df['height_in'].min()) / HEIGHT_INCR)
    _df['weight_bin'] = np.floor((_df['weight_lbs']  - _df['weight_lbs'].min()) / WEIGHT_INCR)
    _df['bmi_bin'] = np.floor((_df['bmi']  - _df['bmi'].min()) / BMI_INCR)
    _df.drop(["standard_size","height_in","weight_lbs","bmi"], axis=1, inplace=True)
    index_df = pd.DataFrame(grouped_df.index)
    _df.set_index('user_id', inplace=True)
    df = index_df.join(_df,on='user_id',how='left')
    return(df.to_dict('records'))

def make_userId2idx_dict(user_attr):
    d = user_attr['user_id'].to_dict()
    d = {v: k for k, v in d.items()}
    return d


class UserStyleMat(object):
    '''
    data structure with convenience functions for matrix factorization.
    '''
    
    def __init__(self, data, user_attrib_dict, userId2idx_dict, style_names):
        
        self.data = data
        self.user_attrib_dict = user_attrib_dict
        self.style_names = style_names
        self.userId2idx_dict = userId2idx_dict
        
        df = pd.DataFrame(user_attrib_dict)
        
        self.n_sizes = df['size_bin'].max()  + 1
        self.n_heights = df['height_bin'].max()  + 1
        self.n_weights = df['weight_bin'].max()  + 1
        self.n_bmis = df['bmi_bin'].max()  + 1
        
    def get_user_params(self, idx):
        return self.user_attrib_dict[idx]
    
    def userid2idx(self, user_id):
        return self.userId2idx_dict[user_id]


def data_pipeline(orders, reviews, user_attr, thin_users, thin_styles, training_prop, testing_prop):
    
    g = prepare_data(orders, reviews, thin_users=thin_users, thin_styles=thin_styles)
    (train_raw, test_raw) = data2sparse(g, training_prop, testing_prop)
    print("TRAINING SHAPE {}; NNZ: {}; Sparsity {:0.4f}%".format(
        train_raw.shape, train_raw.nnz, 100*(1 - (train_raw.nnz / (train_raw.shape[0] * train_raw.shape[1]))))
         )
    print("TESTING SHAPE {}; NNZ: {}; Sparsity {:0.4f}%".format(
        test_raw.shape, test_raw.nnz, 100*(1 - (test_raw.nnz / (test_raw.shape[0] * test_raw.shape[1]))))
         )
        
    user_attrib_dict = make_user_attrib_dict(user_attr, g)
    userId2idx_dict = make_userId2idx_dict(user_attr)
    style_names = list(g.columns)
    
    train = UserStyleMat(train_raw, user_attrib_dict, userId2idx_dict, style_names)
    test = UserStyleMat(test_raw, user_attrib_dict, userId2idx_dict, style_names)
    
    return(train, test)


class MF():

    def __init__(self, training_USM, testing_USM, K, alpha, beta1, beta2, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (UserStyleMat)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        - iterations    : number of iterations to run for
        """
                        
        self.training_USM = training_USM
        self.testing_USM = testing_USM
        self.num_users, self.num_items = training_USM.data.shape
        self.K = K
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.iterations = iterations
        

    def get_feature_vec(self, bin_idx, feature_vec):
        """
        get the feature vector associated with a bin index.  If bin_indx is NaA return zeros
        """
        if np.isnan(bin_idx):
            vec = np.zeros(self.K)
            indicator = 0
        
        else:
            bin_idx = int(bin_idx)
            vec = feature_vec[bin_idx, :]
            indicator = 1
        
        return (vec, indicator)
    
    def set_feature_vec(self, bin_idx, feature_vec, val):
        """
        set the feature vector at bin_idx to val.
        """
        if np.isnan(bin_idx):
            return
        
        else:
            bin_idx = int(bin_idx)
            feature_vec[bin_idx, :] = val
        return
    
    def get_all_feature_vecs(self, user_idx):
        """
        for a given user id, return all feature vectors and cardinality of this set.
        """
        user_dict = self.training_USM.get_user_params(user_idx)
        (size_vec, size_indic) = self.get_feature_vec(user_dict['size_bin'], self.sizes)
        (height_vec, height_indic) = self.get_feature_vec(user_dict['height_bin'], self.heights)
        (weight_vec, weight_indic) = self.get_feature_vec(user_dict['weight_bin'], self.weights)
        (bmi_vec, bmi_indic) = self.get_feature_vec(user_dict['bmi_bin'], self.bmis)
        
        cardinality = size_indic + height_indic + weight_indic + bmi_indic
        cardinality = max(cardinality, 1)
        return(size_vec, height_vec, weight_vec, bmi_vec, cardinality)

    def set_all_feature_vecs(self, user_idx, size_vals, height_vals, weight_vals, bmi_vals):
        """
        set all feature vectors for a given user
        """
        user_dict = self.training_USM.get_user_params(user_idx)
        self.set_feature_vec(user_dict['size_bin'], self.sizes, size_vals)
        self.set_feature_vec(user_dict['height_bin'], self.heights, height_vals)
        self.set_feature_vec(user_dict['weight_bin'], self.weights, weight_vals)
        self.set_feature_vec(user_dict['bmi_bin'], self.bmis, bmi_vals)
        return
    
    def train(self):
        print("Training for k={}; alpha={}; beta1={}; beta2={}".format(self.K, self.alpha, self.beta1, self.beta2))
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.training_USM.data)
        
        # Initialize the additional user latent feature matrices.
        self.sizes = np.random.normal(scale=1./self.K, size=(int(self.training_USM.n_sizes), self.K))
        self.heights = np.random.normal(scale=1./self.K, size=(int(self.training_USM.n_heights), self.K))
        self.weights = np.random.normal(scale=1./self.K, size=(int(self.training_USM.n_weights), self.K))
        self.bmis = np.random.normal(scale=1./self.K, size=(int(self.training_USM.n_bmis), self.K))

        # Create a list of training samples
        self.samples = [
            (i[0][0], i[0][1], i[1])
            for i in self.training_USM.data.items()
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        rmse_train_baseline = self.rmse_baseline(self.training_USM.data)
        rmse_test_baseline = self.rmse_baseline(self.testing_USM.data)
        epoch = 0
        exit_code = 0
        while (epoch < self.iterations) and (exit_code == 0):
            np.random.shuffle(self.samples)
            exit_code = self.sgd()
            rmse_train = self.rmse(self.training_USM.data)
            rmse_test = self.rmse(self.testing_USM.data)
            training_process.append((epoch, rmse_train, rmse_train_baseline, rmse_test, rmse_test_baseline))
            epoch += 1
            if (epoch) % 10 == 0:
                print("Epoch: %d\n" \
                "\ttrain_err = %.4f; train_baseline_err = %.4f\n" \
                      "\ttest_err = %.4f; test_baseline_err = %.4f" % (
                          epoch, rmse_train, rmse_train_baseline, rmse_test, rmse_test_baseline)
                     )

            self.alpha = .9*self.alpha
            
        
        test_errors = [tp[3] for tp in training_process]
        best_test_error = min(test_errors)
        print("BEST TEST: {}".format(best_test_error))
        return (best_test_error, ((self.K, self.alpha, self.beta1, self.beta2)))

    def rmse(self, data):
        """
        A function to compute the total mean square error
        """
        
        error = 0
        n = data.nnz
        for (x, y) in data.keys():
            (size_vec, height_vec, weight_vec, bmi_vec, cardinality) = self.get_all_feature_vecs(x)
            user_vec = cardinality**(-.5)*(size_vec + height_vec + weight_vec + bmi_vec)
            error += pow(data[x, y] - self.get_rating(x,y,user_vec), 2)
        return np.sqrt(error/n)
    
    def rmse_baseline(self, data):
        """
        compute rmse error w.r.t. a baseline predictor that always guess average value.
        """
        error = 0
        n = data.nnz
        mean_nz = np.mean(list(data.values()))
        for (x, y) in data.keys():
            error += pow(data[x, y] - mean_nz, 2)
        return np.sqrt(error/n)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        counter = 0
        for i, j, r in self.samples:
         
            (size_vec, height_vec, weight_vec, bmi_vec, cardinality) = self.get_all_feature_vecs(i)
            user_vec = cardinality**(-.5)*(size_vec + height_vec + weight_vec + bmi_vec)

            # Computer prediction and error
            prediction = self.get_rating(i, j, user_vec)
            e = (r - prediction)
            if(np.isnan(e)):
                print("Terminating SGD: Gradients became badly conditioned " \
                      "increase regularization or decrease stepsize")
                return(1)
        
            self.b_u[i] += self.alpha * (e - self.beta1 * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta1 * self.b_i[j])

            # Update user and item latent feature matrices
            self.Q[j, :] += self.alpha * (e * (self.P[i, :]+user_vec) - self.beta2 * self.Q[j,:])
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta2 * self.P[i,:])
            
            # update user features
            size_vec += self.alpha * (e *self.Q[j, :] - self.beta2 * size_vec)
            height_vec += self.alpha * (e *self.Q[j, :] - self.beta2 * height_vec)
            weight_vec += self.alpha * (e *self.Q[j, :] - self.beta2 * weight_vec)
            bmi_vec += self.alpha * (e *self.Q[j, :] - self.beta2 * bmi_vec)
            
            self.set_all_feature_vecs(i, size_vec, height_vec, weight_vec, bmi_vec)
        return(0)

    def get_rating(self, i, j, user_vec):
        """
        Get the predicted rating of user i and item j
        """           
        prediction = self.b + self.b_u[i] + self.b_i[j] + (self.P[i, :] + user_vec).dot(self.Q[j, :].T)
        return prediction
    
    def predict_for_user(self, user_id):
        """
        Get items sorted by predicted fit for a user in training set.  User bias and 
        """
        i = self.user_id2idx.get(user_id, None)
        if(not i):
            raise ValueError('User Id not found in data')
        else:
            (size_vec, height_vec, weight_vec, bmi_vec, cardinality) = self.get_all_feature_vecs(x)
            user_vec = cardinality**(-.5)*(size_vec + height_vec + weight_vec + bmi_vec)
            
            predictions = self.b + self.b_u[i] + self.b_i + np.matmul(self.Q, (self.P[i, :] + user_vec).T)
            d = {'Score': predictions, 'Style': self.style_names}
            return pd.DataFrame(d).sort_values('Score')

def run_gridsearch(train, test):
    best = np.inf
    ranks = [60,70,80]
    alphas = [.0025, .005, .0075]
    beta1s = [.0008, .001, .0025, .004]
    beta2s = [.04, .06, .08, .1]
    for r in ranks:
        for a in alphas:
            for b1 in beta1s:
                for b2 in beta2s:
                    model = MF(train, test, r, a, b1, b2, 30)
                    (train_err, params) = model.train()
                    if train_err < best:
                        best = train_err
                        best_params = params
                    print("BEST SO FAR {} WITH PARAMS {}".format(best, best_params))
    
    return best, best_params

grid_train, grid_test = data_pipeline(orders, reviews, user_attr, .25, .25, .9, .1)
(best_err, best_params) = run_gridsearch(grid_train, grid_test)
full_train, grid_test = data_pipeline(orders, reviews, user_attr, 1, 1, .9, .1)
model = MF(full_train, full_test, best_params[0], best_params[1],
           best_params[2], best_params[3])
model.train()
