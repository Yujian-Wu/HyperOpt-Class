#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from matplotlib_venn import venn2
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
import json
import pickle
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

import numpy as np
import pandas as pd
from scipy import stats
import copy

import seaborn as sns
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[3]:


set_matplotlib_formats('retina')
plt.rcParams['figure.dpi'] = 250
sns.set_style('dark', {'legend.frameon':True})
np.random.seed(42)


# # Helper functions

# In[4]:


# Distribution of features
def plot_dist(est):
    col = 4
    for i in range(est.shape[1]):
        plt.subplot(est.shape[1]//col + 1, col, i+1)
        plt.title(est.columns[i])
        sns.distplot(est.iloc[:,[i]].values)

    plt.tight_layout()

def myR2(x, y):
    _, _, r_value, _, _ = stats.linregress(x,y)
    return r_value**2

def plotresults(y_train, y_test, predicted_ytrain, predicted_ytest, corlist, mod, composition):
    fig = plt.figure()
    fig.suptitle('%s for %s'%(mod, composition), y = 1.1, fontsize = 20)

    plt.subplot(2,2,1)
    sns.scatterplot(x = y_train.values.ravel(), y = predicted_ytrain)
    ymax1 = max(max(y_train.values.ravel()), max(predicted_ytrain))
    ax = sns.lineplot(x = [0,ymax1], y = [0,ymax1], color = 'red')
    ax.lines[0].set_linestyle(':')
    plt.xlabel('Observed value', fontsize = 5)
    plt.ylabel('Predicted value', fontsize = 5)
    plt.title('Predicted and observed values, training data, $R^2$ = %.4f'
              %myR2(y_train.values.ravel(), predicted_ytrain), fontsize = 7)
    plt.grid()
    
    plt.subplot(2,2,2)
    sns.scatterplot(x = y_test.values.ravel(), y = predicted_ytest)
    ax = sns.lineplot(x = [corlist[0],corlist[1]], y = [corlist[2],corlist[3]], color = 'red')
    ax.lines[0].set_linestyle(':')
    plt.xlabel('Observed value', fontsize = 5)
    plt.ylabel('Predicted value', fontsize = 5)
    plt.title('Predicted and observed values, test data, $R^2$ = %.4f'
              %myR2(y_test.values.ravel(), predicted_ytest), fontsize = 7)
    plt.grid()

    plt.subplot(2,2,3)
    resd1 = y_train.values.ravel() - predicted_ytrain
#     ax3 = plt.bar(x = np.arange(1,len(resd1)+1,1), height = resd1, color = 'black', linewidth = 0)
    ax3 = sns.barplot(x = np.arange(1,len(resd1)+1,1), y = resd1, color = 'black', linewidth = 0)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y,_: '{:.0%}'.format(y/100)))
    plt.title('Residual plot of training', fontsize = 7)
    plt.xlabel('Data point', fontsize = 5)
    plt.ylabel('Residual', fontsize = 5)
    cols = ['green', 'blue', 'yellow', 'orange', 'red']
    for i in range(1,6):
        plt.hlines(y = i, xmin = 1, xmax = len(resd1), linestyles= '--', colors=cols[i-1], linewidth = 0.5, label='%d%%'%i)
        plt.hlines(y = -i, xmin = 1, xmax = len(resd1), linestyles= '--', colors=cols[i-1], linewidth = 0.5)
    plt.legend(frameon=True, loc='lower center', ncol=1, fontsize = 5, bbox_to_anchor = (1.1, 0.5))
    x_tick = np.arange(0, len(resd1), 100)
    plt.xticks(x_tick, x_tick, fontsize = 5)
    plt.grid()

     
    plt.subplot(2,2,4)    
    resd2 = y_test.values.ravel() - predicted_ytest
#     ax4 = plt.bar(x = np.arange(1,len(resd2)+1,1), height = resd2, color = 'black', linewidth = 0)
    ax4 = sns.barplot(x = np.arange(1,len(resd2)+1,1), y = resd2, color = 'black', linewidth = 0)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y,_: '{:.0%}'.format(y/100)))
    for i in range(1,6):
        plt.hlines(y = i, xmin = 1, xmax = len(resd2), linestyles= '--', colors=cols[i-1], linewidth = 0.5, label='%d%%'%i)
        plt.hlines(y = -i, xmin = 1, xmax = len(resd2), linestyles= '--', colors=cols[i-1], linewidth = 0.5)
    plt.legend(frameon=True, loc='lower center', ncol=1, fontsize = 5, bbox_to_anchor = (1.1, 0.5))
    plt.title('Residual plot of test', fontsize = 7)
    plt.xlabel('Data point', fontsize = 5)
    plt.ylabel('Residual', fontsize = 5)
    x_tick2 = np.arange(0, len(resd2), 10)
    plt.xticks(x_tick2, x_tick2, fontsize = 5)
    plt.grid()

    plt.tight_layout()
    plt.show()
    return resd1, resd2

# Residule distribution plot
def res_dist(x, y):
    plt.subplot(1,2,1)
    sns.distplot(x, kde_kws={"lw": 0.5})
    plt.title('Distribution of training residuals')
    plt.xlabel('Residals')
    plt.ylabel('Likelihood')
    
    plt.subplot(1,2,2)
    sns.distplot(y, kde_kws={"lw": 0.5})
    plt.title('Distribution of test residuals')
    plt.xlabel('Residals')
    plt.ylabel('Likelihood')
    
    plt.tight_layout()
    plt.show()


# # Tuning class

# In[5]:


class BaseModels(object):
    def __init__(self, mod, x_train, y_train, x_test, y_test, CV_fold, params = None):
#         params['random_state'] = 42
        self.params = params
        self.model = mod
        self.best = +np.inf
        self.x_train = x_train.astype(float)
        self.y_train = y_train.astype(float)
        self.x_test = x_test.astype(float)
        self.y_test = y_test.astype(float)
        self.fold = CV_fold
        if mod == SVR:
            self.nx_train, self.norm1 = self.normfunc(self.x_train)
            self.ny_train, self.norm2 = self.normfunc(self.y_train)[0].ravel(), self.normfunc(self.y_train)[1]
            self.nx_test = self.norm1.transform(self.x_test)
            self.ny_test = self.norm2.transform(self.y_test)
        self.y_pred = None
        self.train_pred = None
        
#     def fit(self, x, y):
#         return self.model.fit(x, y)
    
#     def predict(self, data):
#         return self.model.predict(data)
    
#     def feature_importance(self, x, y):
#         return self.model.fit(x,y).feature_importances_
    
    def hyperopt_train_test(self, params):
        mod = copy.deepcopy(self.model)
        mod = mod(**params)
        if isinstance(mod, SVR):
            return np.sqrt(-(cross_val_score(mod, self.nx_train, self.ny_train, scoring = 'neg_mean_squared_error', cv = self.fold))).mean()
        else:
            return np.sqrt(-(cross_val_score(mod, self.x_train, self.y_train.values.ravel(), scoring = 'neg_mean_squared_error', cv = self.fold))).mean()

    def f(self, params):
        acc = self.hyperopt_train_test(params)
        if acc < self.best:
            self.best= acc
#         print('New best is', self.best, params)
        return {'loss': acc, 'status': STATUS_OK}
    
    def tuning(self, par_space, iterations):
        trials = Trials()
        best = fmin(self.f, space = par_space, algo = tpe.suggest, max_evals = iterations, trials = trials, verbose = 0)
#         print( best)
        return trials.argmin, trials.best_trial['result']['loss'], trials.trials
    
    def normfunc(self, data):
        norm = StandardScaler()
        return norm.fit_transform(data), norm
    
    def tuningSVR(self, iterations):
        k_list = ['rbf']
        par_spa = {
            'C': hp.uniform('C', 0, 1000),
            'kernel': hp.choice('kernel', k_list),
            'gamma': hp.uniform('gamma', 0, 10),
            'epsilon': hp.uniform('epsilon', 0, 1)
        }
        bp, bp_loss, trials = self.tuning(par_spa, iterations)
        bp['kernel'] = k_list[bp['kernel']]
        return bp, bp_loss, trials
    
    def tuningRF(self, iterations):
        clist = ['mse', 'mae']
        jlist = [-1]
        space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1,10,1)),
        'max_features': scope.int(hp.quniform('max_features', 1,self.x_train.shape[1],1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10,1000,10)),
        'criterion': hp.choice('criterion', clist),
        'random_state': hp.choice('random_state', [42]),
        'n_jobs': hp.choice('n_jobs', jlist)
        }
        bp, bp_loss, trials= self.tuning(space, iterations)

        for i in ['max_depth', 'max_features', 'n_estimators']:
            bp[i] = int(bp[i])

        bp['criterion'] = clist[bp['criterion']]
        bp['random_state'] = 42
        bp['n_jobs'] = -1
        return bp, bp_loss, trials
    
    def tuningLGB(self, iterations):
        type_list = ['gbdt', 'goss', 'dart']
        metric_list = ['rmsle']

        space = {
#         'max_depth': scope.int(hp.uniform('max_depth', 1, 10)),
        'max_bin': scope.int(hp.uniform('max_depth', 2, 50)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10,100, 10)),
#         'boosting_type': hp.choice('boosting_type', type_list),
#         'metric': hp.choice('metric', metric_list),
#         'num_leaves': scope.int(hp.uniform('num_leaves', 30, 100)),
        'num_leaves': scope.int(hp.uniform('num_leaves', 2, 10)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
#         'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 20000, 300000, 20000)),
#         'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 2000, 30000, 2000)),
#         'min_child_samples': scope.int(hp.quniform('min_child_samples', 20, 500, 5)),
        'feature_fraction_seed': scope.int(hp.uniform('feature_fraction_seed', 0, 9)),
        'bagging_seed': scope.int(hp.uniform('bagging_seed', 0, 9)),
        'min_data_in_leaf ': scope.int(hp.uniform('min_data_in_leaf ', 1, 10)),  
        'min_sum_hessian_in_leaf ': scope.int(hp.uniform('min_sum_hessian_in_leaf ', 1, 20)),  
#         'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'bagging_fraction ': hp.uniform('bagging_fraction ', 0.0, 1.0),
        'bagging_freq  ': hp.uniform('bagging_freq  ', 1, 10),
#         'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#         'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
        'feature_fraction': hp.uniform('feature_fraction', 0, 1)
        }

        bp, bp_loss, trials = self.tuning(space, iterations)
    
        for i in ['feature_fraction_seed', 'n_estimators', 'num_leaves', 'bagging_seed', 'max_bin', 'min_data_in_leaf', 'min_sum_hessian_in_leaf']:
            bp[i] = int(bp[i])

#         bp['boosting_type'] = type_list[bp['boosting_type']]
#         bp['metric'] = metric_list[bp['metric']]
        return bp, bp_loss, trials
    
    def mySVR_prediction(self, params):
        mod = copy.deepcopy(self.model)
        svr = mod(**params).fit(self.nx_train, self.ny_train)
        # on test data
        pred = svr.predict(self.nx_test)
        y_pred = self.norm2.inverse_transform(pred)
        # on trainning data
        pred2 = svr.predict(self.nx_train)
        y_pred_train = self.norm2.inverse_transform(pred2)
    
        return y_pred, y_pred_train, svr, mean_squared_error(pred, self.ny_test)
    
    def myRF_prediction(self, params):
        mod = copy.deepcopy(self.model)
        RF = mod(**params).fit(self.x_train, self.y_train.values.ravel())
        # on test data
        y_pred = RF.predict(self.x_test)
        # on trainning data
        y_pred2 = RF.predict(self.x_train)

        return y_pred, y_pred2, RF, mean_squared_error(y_pred, self.y_test.values.ravel())
    
    def myLGB_prediction(self, params):
        mod = copy.deepcopy(self.model)
        mylgb = mod(**params).fit(self.x_train, self.y_train.values.ravel())
        # on test data
        y_pred = mylgb.predict(self.x_test)
        # on trainning data
        y_pred2 = mylgb.predict(self.x_train)

        return y_pred, y_pred2, mylgb, mean_squared_error(y_pred, self.y_test.values.ravel())
    
    def pred_results(self, params, mod_name, title, _range):
        if self.model == SVR:
            self.y_pred, self.train_pred, mod1, _ = self.mySVR_prediction(params=params)
        elif self.model == RandomForestRegressor:
            self.y_pred, self.train_pred, mod1, _ = self.myRF_prediction(params = params)
        elif self.model == LGBMRegressor:
            self.y_pred, self.train_pred, mod1, _ = self.myLGB_prediction(params = params)

        o1_res1, o1_res2 = plotresults(self.y_train, self.y_test, self.train_pred, self.y_pred, _range, mod_name, title)
        res_dist(o1_res1, o1_res2)
        
        l_list = np.concatenate((np.arange(1,5.5,0.5), np.arange(6, 11, 1), np.array([10])))

        summary_l = []
        thre_l = []
        percentage = []
        for i in l_list[:-1]:
            thre_l.append('<= '+str(i)+'%')
            num = sum(np.abs(self.y_pred - self.y_test.values.ravel()) <= i)
            summary_l.append(num)
            percentage.append(str(np.round((num/len(self.y_pred))*100, 2)) + '%')
        thre_l.append('>= '+str(l_list[-1])+'%')
        thre_l.append('Total')
        num2 = sum(np.abs(self.y_pred - self.y_test.values.ravel()) > l_list[-1])
        summary_l.append(num2)
        summary_l.append(len(self.y_pred))
        percentage.append(str(np.round((num2/len(self.y_pred)*100),2)) + '%')
        percentage.append('100%')

        df = pd.DataFrame({'Count':summary_l, 'Percentage':percentage, 'Threshold': thre_l})
        df = df.set_index('Threshold').rename_axis(None)
        display(df.T)
        return o1_res1, o1_res2, self.y_test, self.y_pred


# # 1. Datasets

# ##  1.1 Training sets (decolored)

# In[6]:


dec_feat = pd.read_csv('C:/Users/YWu42/Desktop/Data/Compositions_prediction/Decolored_datasets/DFAOD_features_baseSubtracted_AllCH_v3_472_decolor.csv')
dec_feat2 = dec_feat.drop([i for i in dec_feat.columns if ('X' not in i) & (i.isdigit())], axis = 1)

# Final training dataset
X_train = dec_feat2.astype(float)

training_resp = pd.read_csv('C:/Users/YWu42/Desktop/Data/Compositions_prediction/DFAOD_features_Composition_selected.csv')

# Training reponses
y_train1 = training_resp[['C1_wt']]
y_train2 = training_resp[['C3_5_wt']]
y_train3 = training_resp[['C6plus_wt']]
y_train4 = training_resp[['CO2_wt']]

y_train5 = training_resp[['C2_wt']]
y_train6 = training_resp[['C3_wt']]
y_train7 = training_resp[['C4_wt']]
y_train8 = training_resp[['C5_wt']]


# In[7]:


X_train.head()


# In[8]:


training_ind = training_resp['Type'].values.ravel()
X_train['Type'] = training_ind


# ## Colored data

# In[69]:


training_feat_col = pd.read_csv('C:/Users/YWu42/Desktop/Data/Compositions_prediction/DFAOD_features_baseSubtracted_v2.csv')


# In[143]:


training_feat_col.head()


# #### plot features

# In[140]:


indices = training_resp['Type']


# In[152]:


def plot_feat(df, indices, term):
    dec_feat = df[indices == term]
    for i in range(dec_feat.shape[0]):
        plt.plot(dec_feat.iloc[i, 12:-2].values.ravel())
    plt.xticks(list(range(11)), dec_feat.columns[12:-2], rotation = 45, Fontsize = 3)
    plt.xlabel('OD Channels', Fontsize = 3)
    plt.yticks(Fontsize = 3)
    plt.ylabel('OD Values', Fontsize = 3)
    plt.grid()
    plt.title('Decolored '+ term, Fontsize = 5)
    
def plot_feat2(df, indices, term):
    dec_feat = df[indices == term]
    for i in range(dec_feat.shape[0]):
        plt.plot(dec_feat.iloc[i, 5:-3].values.ravel())
    plt.xticks(list(range(11)), dec_feat.columns[5:-3], rotation = 45, Fontsize = 3)
    plt.xlabel('OD Channels', Fontsize = 3)
    plt.yticks(Fontsize = 3)
    plt.ylabel('OD Values', Fontsize = 3)
    plt.grid()
    plt.title('Colored ' + term, Fontsize = 5)


# In[153]:


plt.subplot(3, 2, 2)
plot_feat(dec_feat, indices, 'Oil')
plt.subplot(3, 2, 4)
plot_feat(dec_feat, indices, 'GC')
plt.subplot(3, 2, 6)
plot_feat(dec_feat, indices, 'Gas')

plt.subplot(3, 2, 1)
plot_feat2(training_feat_col, indices, 'Oil')
plt.subplot(3, 2, 3)
plot_feat2(training_feat_col, indices, 'GC')
plt.subplot(3, 2, 5)
plot_feat2(training_feat_col, indices, 'Gas')

plt.tight_layout()


# ## 1.2 Test sets (decolored)

# In[9]:


# def getTest(X_train, file_dir):
#     test_set = pd.read_csv(file_dir)
#     test_set.columns = ['X'+i if i.isdigit() else i for i in test_set.columns]

#     temp_set = pd.DataFrame()

#     for i in X_train.columns:
#         if i in test_set.columns:
#             temp_set[[i]] = test_set[[i]]

#     if (X_train.columns == temp_set.columns).all():
#         X_test = temp_set
        
#     y_test1 = test_set[['C1_wt']]
#     y_test2 = test_set[['C3_5_wt']]
#     y_test3 = test_set[['C6plus_wt']]
#     y_test4 = test_set[['CO2_wt']]

#     y_test5 = test_set[['C2_wt']]
#     y_test6 = test_set[['C3_wt']]
#     y_test7 = test_set[['C4_wt']]
#     y_test8 = test_set[['C5_wt']]
#     return X_test, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8


# In[9]:


test_feat = pd.read_table('C:/Users/YWu42/Desktop/Data/Compositions_prediction/Decolored_datasets/OD_PT_FISO_Test_24CH_103_decolor.txt',header = 0)
# Rename columns to match them with training data
test_feat.columns = ['X'+i if i.isdigit() else i for i in test_feat.columns]
X_test = copy.copy(test_feat[[i for i in X_train.columns[:-1]]])

# Test reponses
test_resp = pd.read_table('C:/Users/YWu42/Desktop/Data/Compositions_prediction/Decolored_datasets/OD_FISO_Test_24CH_Composition_103.txt')
y_test1 = test_resp[['C1']]
y_test2 = test_resp[['C35']]
y_test3 = test_resp[['C6+']]
y_test4 = test_resp[['CO2']]

y_test5 = test_resp[['C2']]
y_test6 = test_resp[['C3']]
y_test7 = test_resp[['C4']]
y_test8 = test_resp[['C5']]


# In[10]:


test_feat.head()


# In[11]:


test_ind = []
for i in test_feat['Sample'].values.ravel():
    if 'Oil' in i:
        test_ind.append('Oil')
    elif 'Cond' in i:
        test_ind.append('GC')
    elif 'gas' in i:
        test_ind.append('Gas')
test_ind = np.array(test_ind)
X_test['Type'] = test_ind
# len(test_ind)


# ## 1.3 Training sets (colored)

# In[64]:


# Traning feature set -- 472
def color_train():
    training_feat = pd.read_csv('C:/Users/YWu42/Desktop/Data/Compositions_prediction/DFAOD_features_baseSubtracted_v2.csv')
    # Traning response set -- 472
    training_resp = pd.read_csv('C:/Users/YWu42/Desktop/Data/Compositions_prediction/DFAOD_features_Composition_selected.csv')

    X_train = training_feat.drop(['X680', 'X815', 'X1070','X1290','X1445', 'Type'], axis = 1).astype(float)

    y_train1 = training_resp[['C1_wt']]
    y_train2 = training_resp[['C3_5_wt']]
    y_train3 = training_resp[['C6plus_wt']]
    y_train4 = training_resp[['CO2_wt']]

    y_train5 = training_resp[['C2_wt']]
    y_train6 = training_resp[['C3_wt']]
    y_train7 = training_resp[['C4_wt']]
    y_train8 = training_resp[['C5_wt']]
    return X_train, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7, y_train8

def color_test(X_train):
    test_set = pd.read_csv('C:/Users/YWu42/Desktop/Data/Compositions_prediction/TestCases_KaiFISO_withGases_DFAnmOD_addedCH_v2.csv')
    test_set.columns = ['X'+i if i.isdigit() else i for i in test_set.columns]

    temp_set = pd.DataFrame()

    for i in X_train.columns:
        if i in test_set.columns:
            temp_set[[i]] = test_set[[i]]

    if (X_train.columns == temp_set.columns).all():
        X_test = temp_set
        
    y_test1 = test_set[['C1_wt']]
    y_test2 = test_set[['C3_5_wt']]
    y_test3 = test_set[['C6plus_wt']]
    y_test4 = test_set[['CO2_wt']]

    y_test5 = test_set[['C2_wt']]
    y_test6 = test_set[['C3_wt']]
    y_test7 = test_set[['C4_wt']]
    y_test8 = test_set[['C5_wt']]
    return X_test, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6, y_test7, y_test8


# In[65]:


train_col = color_train()
test_col = color_test(train_col[0])


# In[97]:


test_list = getTest(train_col[0], 'C:/Users/YWu42/Desktop/Data/Compositions_prediction/TestCases_KaiFISO_withGases_DFAnmOD_addedCH_2014_2015.csv')


# # 2. Feature engineering

# ## 2.1 C6plus

# In[12]:


def C6feat(X_train, X_test):
    ntrain = X_train.shape[0]
    total = pd.concat((X_train, X_test), axis = 0)
    # Feature engineering
    total2 = total.iloc[:,:-3].subtract(total['X1671'], axis = 0).drop('X1671', axis = 1, inplace = False)
    total2[['P_psi', 'T_C']] = total[['P_psi', 'T_C']]
    total2['X1671ratio'] = total['X1671']/total.iloc[:,3:-2].sum(axis = 1)
    total2['X1650ratio'] = total['X1650']/total.iloc[:,3:-2].sum(axis = 1)
    total2['C6ratio'] = (total['X1725']+total['X1760'])/total[total.columns.difference(['X1725', 'X1760', 'P_psi','T_C','Type'])].sum(axis=1)
#     # categorical temperature
#     total2['T_cat'] = np.zeros(total2.shape[0])
#     total2.loc[(total2.T_C>=25)&(total2.T_C<=42.5), ['T_cat']] = '1'
#     total2.loc[(total2.T_C>60)&(total2.T_C<=95), ['T_cat']] = '2'
#     total2.loc[(total2.T_C>=100)&(total2.T_C<=130), ['T_cat']] = '3'
#     total2.loc[(total2.T_C>147.5)&(total2.T_C<=200), ['T_cat']] = '4'
    
#     # categorical pressure
#     total2['P_cat'] = np.zeros(total2.shape[0])
#     total2.loc[(total2.P_psi>=1000)&(total2.P_psi<=5800), ['P_cat']] = '1'
#     total2.loc[(total2.P_psi>5800)&(total2.P_psi<=10600), ['P_cat']] = '2'
#     total2.loc[(total2.P_psi>10600)&(total2.P_psi<=15400), ['P_cat']] = '3'
#     total2.loc[(total2.P_psi>15400)&(total2.P_psi<=22600), ['P_cat']] = '4'
#     total2.loc[(total2.P_psi>22600)&(total2.P_psi<=25000), ['P_cat']] = '5'
    
    total2['Type'] = total['Type']
    total2['PTratio'] = total2['P_psi']/total2['T_C']
    total2 = pd.get_dummies(total2)
    
    X_train = total2.iloc[:ntrain]
    X_test = total2.iloc[ntrain:]
    return X_train, X_test


# In[13]:


X_train2, X_test2 = C6feat(X_train, X_test)


# In[14]:


X_test2.head()


# ## 2.2 C1

# In[72]:


def C1feat(X_train, X_test):
    ntrain = X_train.shape[0]
    total = pd.concat((X_train, X_test), axis = 0)
    # Feature engineering
    total2 = total.iloc[:,:-3].subtract(total['X1671'], axis = 0).drop('X1671', axis = 1, inplace = False)
    total2[['P_psi', 'T_C']] = total[['P_psi', 'T_C']]
    total2['X1671ratio'] = total['X1671']/total.iloc[:,3:-2].sum(axis = 1)
    total2['X1650ratio'] = total['X1650']/total.iloc[:,3:-2].sum(axis = 1)
#     total2['C1ratio'] = (total['X1650']+total['X1671'])/total[total.columns.difference(['X1650', 'X1671', 'P_psi','T_C','Type'])].sum(axis=1)
    total2['Type'] = total['Type']
#     total2['PTratio'] = total2['P_psi']/total2['T_C']
    total2 = pd.get_dummies(total2)
    
    X_train = total2.iloc[:ntrain]
    X_test = total2.iloc[ntrain:]
    return X_train, X_test


# In[73]:


X_trainC1, X_testC1 = C1feat(X_train, X_test)


# # 4. SVR

# In[28]:


def tab_func(y_pred, y_test): 
    l_list = np.concatenate((np.arange(1,5.5,0.5), np.arange(6, 11, 1), np.array([10])))

    summary_l = []
    thre_l = []
    percentage = []
    for i in l_list[:-1]:
        thre_l.append('<= '+str(i)+'%')
        num = sum(np.abs(y_pred - y_test.values.ravel()) <= i)
        summary_l.append(num)
        percentage.append(str(np.round((num/len(y_pred))*100, 2)) + '%')
    thre_l.append('>= '+str(l_list[-1])+'%')
    thre_l.append('Total')
    num2 = sum(np.abs(y_pred - y_test.values.ravel()) > l_list[-1])
    summary_l.append(num2)
    summary_l.append(len(y_pred))
    percentage.append(str(np.round((num2/len(y_pred)*100),2)) + '%')
    percentage.append('100%')

    df = pd.DataFrame({'Count':summary_l, 'Percentage':percentage, 'Threshold': thre_l})
    df = df.set_index('Threshold').rename_axis(None)
    display(df.T)


# ## 4.1 C6+ decolored

# In[114]:


C6s = BaseModels(SVR, X_train2, y_train3/100, X_test2, y_test3/100, 5)


# In[115]:


C6s_params, C6s_loss, C6s_track = C6s.tunningSVR(5000)


# In[116]:


_,_, a6s, b6s = C6s.pred_results(C6s_params, 'SVR', 'C6plus_wt', [0, 100, 0, 100])


# In[118]:


tab_func(b6s*100, y_test3)


# In[116]:


# # Save the parameters
# with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C6_SVR_dec_5000.pickle','wb') as f:
#     pickle.dump([C6s_params, C6s_loss, C6s_track], f)


# In[46]:


# with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C6_SVR_dec_5000.pickle','rb') as f:
#     C6s_params, C6s_loss, C6s_track = pickle.load(f)


# ## 4.2 C1 decolored

# In[211]:


C1 = BaseModels(SVR, X_train, y_train1, X_test, y_test1, 5)


# In[14]:


C1_params, C1_loss, C1_track = C1.tunningSVR(5000)


# In[224]:


r1, r2, a1, b1 = C1.pred_results(C1_params, 'SVR', 'C1_wt', [0, 100, 0, 100])


# In[16]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C1_SVR_dec_5000.pickle','wb') as f:
    pickle.dump([C1_params, C1_loss, C1_track], f)


# #### C1 engineered features

# In[51]:


C1_s = BaseModels(SVR, X_trainC1, y_train1, X_testC1, y_test1, 5)


# In[52]:


C1_params_s, C1_loss_s, C1_track_s = C1_s.tunningSVR(5000)


# In[53]:


r1s, r2s, a1s, b1s = C1_s.pred_results(C1_params_s, 'SVR', 'C1_wt', [0, 100, 0, 100])


# ### Separate Residuals

# In[268]:


sep_res_plot(a1, b1)


# ## 4.3 C2 decolored

# In[17]:


C2 = BaseModels(SVR, X_train, y_train5, X_test, y_test5, 5)
C2_params, C2_loss, C2_track = C2.tunningSVR(5000)


# In[269]:


_,_, a2, b2 = C2.pred_results(C2_params, 'SVR', 'C2_wt', [0, 20, 0, 20])


# In[19]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C2_SVR_dec_5000.pickle','wb') as f:
    pickle.dump([C2_params, C2_loss, C2_track], f)


# In[270]:


sep_res_plot(a2, b2)


# ## 4.4 C3 decolored

# In[20]:


C3 = BaseModels(SVR, X_train, y_train6, X_test, y_test6, 5)
C3_params, C3_loss, C3_track = C3.tunningSVR(5000)


# In[49]:


_,_, a3, b3 = C3.pred_results(C3_params, 'SVR', 'C3_wt', [0, 10, 0, 10])


# In[22]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C3_SVR_dec_5000.pickle','wb') as f:
    pickle.dump([C3_params, C3_loss, C3_track], f)


# In[271]:


sep_res_plot(a3, b3)


# ## 4.5 C4 decolored

# In[23]:


C4 = BaseModels(SVR, X_train, y_train7, X_test, y_test7, 5)
C4_params, C4_loss, C4_track = C4.tunningSVR(5000)


# In[50]:


_,_, a4, b4 = C4.pred_results(C4_params, 'SVR', 'C4_wt', [0, 10, 0, 10])


# In[25]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C4_SVR_dec_5000.pickle','wb') as f:
    pickle.dump([C4_params, C4_loss, C4_track], f)


# In[272]:


sep_res_plot(a4, b4)


# ## 4.6 C5 decolored

# In[26]:


C5 = BaseModels(SVR, X_train, y_train8, X_test, y_test8, 5)
C5_params, C5_loss, C5_track = C5.tunningSVR(5000)


# In[52]:


_,_, a5, b5 = C5.pred_results(C5_params, 'SVR', 'C5_wt', [0, 10, 0, 10])


# In[28]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/C5_SVR_dec_5000.pickle','wb') as f:
    pickle.dump([C5_params, C5_loss, C5_track], f)


# In[273]:


sep_res_plot(a5, b5)


# ## 4.7 $CO_2$ decolored

# In[29]:


CO = BaseModels(SVR, X_train, y_train4, X_test, y_test4, 5)
CO_params, CO_loss, CO_track = CO.tunningSVR(5000)


# In[53]:


_,_, ao, bo = CO.pred_results(CO_params, 'SVR', '$CO_2$_wt', [0, 10, 0, 10])


# In[31]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190318 Feature_engineering/CO2_SVR_dec_5000.pickle','wb') as f:
    pickle.dump([CO_params, CO_loss, CO_track], f)


# In[274]:


sep_res_plot(ao, bo)


# # 5. Training-validation-test

# In[15]:


def train_val_split(X_train, y_train3, training_ind):
    new_xtrain = pd.DataFrame()
    new_xval = pd.DataFrame()
    new_ytrain = pd.DataFrame()
    new_yval = pd.DataFrame()

    for i in set(training_ind):
        temp_xtrain, temp_xval, temp_ytrain, temp_yval = train_test_split(X_train[training_ind == i], y_train3[training_ind == i], random_state = 42, shuffle = True)
        new_xtrain = pd.concat((new_xtrain, temp_xtrain), axis = 0)
        new_xval = pd.concat((new_xval, temp_xval), axis = 0)
        new_ytrain = pd.concat((new_ytrain, temp_ytrain), axis = 0)
        new_yval = pd.concat((new_yval, temp_yval), axis = 0)
    return new_xtrain, new_xval, new_ytrain, new_yval


# In[16]:


# C6+
nX_train, nX_val, ny_train3, ny_val3 = train_val_split(X_train2, y_train3, training_ind)


# In[17]:


nX_train.head()


# In[74]:


#C1
nX_train, nX_val, ny_train1, ny_val1 = train_val_split(X_trainC1, y_train1, training_ind)


# ### helpers

# In[22]:


def getSVRdf(term):
    par_file = 'C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/%s.pickle'%term

    with open(par_file, 'rb') as f:
        f_list = pickle.load(f)

    loss_list = [i['result']['loss'] for i in f_list[2]]
    c_list = [i['misc']['vals']['C'][0] for i in f_list[2]]
    e_list = [i['misc']['vals']['epsilon'][0] for i in f_list[2]]
    g_list = [i['misc']['vals']['gamma'][0] for i in f_list[2]]
    iterations = [i['tid'] for i in f_list[2]] 
    
    param_df = pd.DataFrame({
        'C': c_list,
        'epsilon': e_list,
        'gamma': g_list,
        'training_loss': loss_list,
        'ID': iterations
    })
    return param_df

def test_loss(term, mod, X_train, y_train1, X_test, y_test1):
    param_df = getSVRdf(term)
    mod = BaseModels(mod, X_train, y_train1, X_test, y_test1, 5)
    test_loss = []
    for i in range(param_df.shape[0]):
        if (i+1) % 1000 == 0:
            print(i)
        cols = param_df.columns[:-2]
        temp_par = {cols[j]: param_df.loc[i, cols[j]] for j in range(len(cols))}
        test_loss.append(mod.mySVR_prediction(temp_par)[3])
    param_df['test_loss'] = test_loss
    
    test_par = {
        'C':param_df.loc[(param_df['test_loss'] == min(param_df['test_loss'])), ['C']].values.ravel()[0],
        'epsilon':param_df.loc[(param_df['test_loss'] == min(param_df['test_loss'])), ['epsilon']].values.ravel()[0],
        'gamma':param_df.loc[(param_df['test_loss'] == min(param_df['test_loss'])), ['gamma']].values.ravel()[0],
        'kernel': 'rbf'
    }
    return param_df, test_par


# In[23]:


def getRFdf(term):
    par_file = 'C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/%s.pickle'%term

    with open(par_file, 'rb') as f:
        f_list = pickle.load(f)

    loss_list = [i['result']['loss'] for i in f_list[2]]
    d_list = [int(i['misc']['vals']['max_depth'][0]) for i in f_list[2]]
    feat_list = [int(i['misc']['vals']['max_features'][0]) for i in f_list[2]]
    e_list = [int(i['misc']['vals']['n_estimators'][0]) for i in f_list[2]]
    c_list = [['mse', 'mae'][i['misc']['vals']['criterion'][0]] for i in f_list[2]]
    iterations = [i['tid'] for i in f_list[2]] 
    
    param_df = pd.DataFrame({
        'criterion': c_list,
        'max_depth': d_list,
        'max_features': feat_list,
        'n_estimators': e_list,
        'training_loss': loss_list,
        'ID': iterations
    })
    return param_df

def test_RFloss(term, mod, X_train, y_train1, X_test, y_test1):
    param_df = getRFdf(term)
    mod = BaseModels(mod, X_train, y_train1, X_test, y_test1, 5)
    test_loss = []
    for i in range(param_df.shape[0]):
        if (i+1) % 1000 == 0:
            print(i)
        cols = param_df.columns[:-2]
        temp_par = {cols[j]: param_df.loc[i, cols[j]] for j in range(len(cols))}
        temp_par['random_state'] = 42
        temp_par['n_jobs'] = -1
        test_loss.append(mod.myRF_prediction(temp_par)[3])
    param_df['test_loss'] = test_loss
    
    min_df = param_df.loc[param_df['test_loss'] == min(param_df['test_loss'])]
    
    test_par = {
        min_df.columns[i]: min_df[[min_df.columns[i]]].values.ravel()[0] for i in range(min_df.shape[1]-3)
    }
    test_par['random_state'] = 42
    test_par['n_jobs'] = -1
    return param_df, test_par


# In[166]:


def getLGBMdf(term):
    par_file = 'C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/%s.pickle'%term

    with open(par_file, 'rb') as f:
        f_list = pickle.load(f)

    loss_list = [i['result']['loss'] for i in f_list[2]]
    d_list = [int(i['misc']['vals']['max_depth'][0]) for i in f_list[2]]
    e_list = [int(i['misc']['vals']['n_estimators'][0]) for i in f_list[2]]
    c_list = [i['misc']['vals']['colsample_by_tree'][0] for i in f_list[2]]
    b_list = [['gbdt', 'goss', 'dart'][i['misc']['vals']['boosting_type'][0]] for i in f_list[2]]
    l_list = [i['misc']['vals']['learning_rate'][0] for i in f_list[2]]
    m_list = ['rmsle' for i in f_list[2]]
    mc_list = [int(i['misc']['vals']['min_child_samples'][0]) for i in f_list[2]]
    n_list = [int(i['misc']['vals']['num_leaves'][0]) for i in f_list[2]]
    s_list = [int(i['misc']['vals']['subsample_for_bin'][0]) for i in f_list[2]]
    ra_list = [i['misc']['vals']['reg_alpha'][0] for i in f_list[2]]
    rl_list = [i['misc']['vals']['reg_lambda'][0] for i in f_list[2]]
    
    iterations = [i['tid'] for i in f_list[2]] 
    
    param_df = pd.DataFrame({
         'boosting_type': b_list,
         'colsample_by_tree': c_list,
         'learning_rate': l_list,
         'max_depth': d_list,
         'metric': m_list,
         'min_child_samples': mc_list,
         'n_estimators': e_list,
         'num_leaves': n_list,
         'reg_alpha': ra_list,
         'reg_lambda': rl_list,
         'subsample_for_bin': s_list,
         'training_loss': loss_list,
         'ID': iterations
    })
    return param_df

def test_LGBMloss(term, mod, X_train, y_train1, X_test, y_test1):
    param_df = getLGBMdf(term)
    mod = BaseModels(mod, X_train, y_train1, X_test, y_test1, 5)
    test_loss = []
    for i in range(param_df.shape[0]):
        if (i+1) % 1000 == 0:
            print(i)
        cols = param_df.columns[:-2]
        temp_par = {cols[j]: param_df.loc[i, cols[j]] for j in range(len(cols))}
        temp_par['random_state'] = 42
        temp_par['n_jobs'] = -1
        test_loss.append(mod.myRF_prediction(temp_par)[3])
    param_df['test_loss'] = test_loss
    
    min_df = param_df.loc[param_df['test_loss'] == min(param_df['test_loss'])]
    
    test_par = {
        min_df.columns[i]: min_df[[min_df.columns[i]]].values.ravel()[0] for i in range(min_df.shape[1]-3)
    }
    test_par['random_state'] = 42
    test_par['n_jobs'] = -1
    return param_df, test_par


# ## C6 SVR

# ### Tuning training parameters

# In[227]:


C6s = BaseModels(SVR, nX_train, np.log1p(ny_train3), nX_val, np.log1p(ny_val3), 5)


# In[229]:


C6s_params, C6s_loss, C6s_track = C6s.tuningSVR(5000)


# In[230]:


_,_, a6s, b6s = C6s.pred_results(C6s_params, 'SVR', 'C6plus_wt', [0, 100, 0, 100])


# In[231]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_SVR_dec_5000_type_dummy_C6ratio_ptratio_logres.pickle','wb') as f:
    pickle.dump([C6s_params, C6s_loss, C6s_track], f)


# ### Best on Validation

# In[232]:


# decorlor
C6_tab, C6_bp = test_loss('C6_train_val_SVR_dec_5000_type_dummy_C6ratio_ptratio_logres', SVR, nX_train, np.log1p(ny_train3), nX_val, np.log1p(ny_val3))


# In[233]:


# decolor
C6val = BaseModels(SVR, nX_train, np.log1p(ny_train3), nX_val, np.log1p(ny_val3), 5)
_,_,_,_ = C6val.pred_results(C6s_params, 'SVR', 'C6_wt', [0, 100, 0, 100])


# In[234]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_SVR_dec_5000_best_val_type_dummy_C6ratio_ptratio_logres.pickle','wb') as f:
    pickle.dump([C6_tab, C6_bp], f)


# In[195]:


C6_bpp


# ### Try on test data

# In[135]:


with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_SVR_dec_5000_best_val_type_dummy_C6ratio_ptratio.pickle','rb') as f:
    C6_tab, C6_bp = pickle.load(f)


# In[235]:


C6test = BaseModels(SVR, nX_train, np.log1p(ny_train3), X_test2, np.log1p(y_test3), 5)
_,_, _, z = C6test.pred_results(C6_bp, 'SVR', 'C6_wt', [0, 100, 0, 100])


# ## C1 SVR

# ### Tuning

# In[43]:


C1s = BaseModels(SVR, nX_train, ny_train1, nX_val, ny_val1, 5)


# In[44]:


C1s_params, C1s_loss, C1s_track = C1s.tunningSVR(5000)


# In[45]:


_,_, a1s, b1s = C1s.pred_results(C1s_params, 'SVR', 'C1_wt', [0, 100, 0, 100])


# In[46]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C1_train_val_SVR_dec_5000_type_dummy_C1ratio_ptratio_nosub.pickle','wb') as f:
    pickle.dump([C1s_params, C1s_loss, C1s_track], f)


# ### Best on validation

# In[47]:


# decorlor
C1_tab, C1_bp = test_loss('C1_train_val_SVR_dec_5000_type_dummy_C1ratio_ptratio_nosub', SVR, nX_train, ny_train1, nX_val, ny_val1)


# In[88]:


# decolor
C1val = BaseModels(SVR, nX_train, ny_train1, nX_val, ny_val1, 5)
_,_,_, _ = C1val.pred_results(C1_bp, 'SVR', 'C1_wt', [0, 100, 0, 100])


# In[49]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C1_train_val_SVR_dec_5000_best_val_type_dummy_C1ratio_ptratio_nosub.pickle','wb') as f:
    pickle.dump([C1_tab, C1_bp], f)


# ### Test

# In[89]:


C1test = BaseModels(SVR, nX_train, ny_train1, X_testC1, y_test1, 5)
_,_, _, nosub_pred = C1test.pred_results(C1_bp, 'SVR', 'C1_wt', [0, 100, 0, 100])


# ## C6 RF

# ### Tuning training

# In[108]:


C6rf = BaseModels(RandomForestRegressor, nX_train, ny_train3, nX_val, ny_val3, 5)


# In[109]:


C6r_params, C6r_loss, C6r_track = C6rf.tunningRF(200)


# In[110]:


_,_, a6r, b6r = C6rf.pred_results(C6r_params, 'Random Forest', 'C6plus_wt', [0, 100, 0, 100])


# In[127]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_RF_dec_200_type_dummy_C6ratio_ptratio.pickle','wb') as f:
    pickle.dump([C6r_params, C6r_loss, C6r_track], f)


# ### Best on validation

# In[133]:


# C6r_tab, C6r_bp = test_RFloss('C6_train_val_RF_dec_200_type_dummy_C6ratio_ptratio', RandomForestRegressor, nX_train, ny_train3, nX_val, ny_val3)


# In[113]:


C6v = BaseModels(RandomForestRegressor, nX_train, ny_train3, nX_val, ny_val3, 5)
_,_,_,_ = C6v.pred_results(C6r_bp, 'Random Forest', 'C6plus_wt', [0, 100, 0, 100])


# In[131]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_RF_dec_200_type_dummy_C6ratio_ptratio_bestval.pickle','wb') as f:
    pickle.dump([C6r_tab, C6r_bp], f)


# In[196]:


C6r_bp


# ### Try on test

# In[130]:


with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_svr_dec_200_type_dummy_C6ratio_ptratio_bestval.pickle','rb') as f:
    C6r_tab, C6r_bp = pickle.load(f)


# In[132]:


C6rtest = BaseModels(RandomForestRegressor, nX_train, ny_train3, X_test2, y_test3, 5)
_,_, _, _ = C6rtest.pred_results(C6r_bp, 'Random Forest', 'C6_wt', [0, 100, 0, 100])


# ## C6 LGBM

# ### Tuning training data

# In[18]:


C6gbm = BaseModels(LGBMRegressor, nX_train, ny_train3, nX_val, ny_val3, 5)


# In[ ]:


C6gbm_params, C6gbm_loss, C6gbm_track = C6gbm.tuningLGB(40)


# In[ ]:


_,_, a6gbm, b6gbm = C6gbm.pred_results(C6gbm_params, 'Light GBM', 'C6plus_wt', [0, 100, 0, 100])


# In[190]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_LGBM_dec_1000_type_dummy_C6ratio_ptratio.pickle','wb') as f:
    pickle.dump([C6gbm_params, C6gbm_loss, C6gbm_track], f)


# ### Best on validation

# In[191]:


C6lgbm_tab, C6lgbm_bp = test_LGBMloss('C6_train_val_LGBM_dec_1000_type_dummy_C6ratio_ptratio', LGBMRegressor, nX_train, ny_train3, nX_val, ny_val3)


# In[192]:


# decolor
C6lgbmval = BaseModels(LGBMRegressor, nX_train, ny_train3, nX_val, ny_val3, 5)
_,_,_,_ = C6lgbmval.pred_results(C6lgbm_bp, 'LightGBM', 'C6_wt', [0, 100, 0, 100])


# In[193]:


# Save the parameters
with open('C:/Users/YWu42/Desktop/Data/Compositions_prediction/20190321 TrainValSplit/C6_train_val_LGBM_dec_1000_type_dummy_C6ratio_ptratio_bestval.pickle','wb') as f:
    pickle.dump([C6lgbm_tab, C6lgbm_bp], f)


# In[197]:


C6lgbm_bp


# ### Test

# In[194]:


C6gbmtest = BaseModels(LGBMRegressor, nX_train, ny_train3, X_test2, y_test3, 5)
_,_, _, z2 = C6gbmtest.pred_results(C6lgbm_bp, 'LightGBM', 'C6_wt', [0, 100, 0, 100])


# # Further analysis

# In[120]:


def new_resplot(y_test, predicted_ytest, term):
    cols = ['green', 'blue', 'yellow', 'orange', 'red']
    resd2 = y_test - predicted_ytest
    #     ax4 = plt.bar(x = np.arange(1,len(resd2)+1,1), height = resd2, color = 'black', linewidth = 0)
    ax4 = sns.barplot(x = np.arange(1,len(resd2)+1,1), y = resd2, color = 'black', linewidth = 0)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda y,_: '{:.0%}'.format(y/100)))
    for i in range(1,6):
        plt.hlines(y = i, xmin = 1, xmax = len(resd2), linestyles= '--', colors=cols[i-1], linewidth = 0.5, label='%d%%'%i)
        plt.hlines(y = -i, xmin = 1, xmax = len(resd2), linestyles= '--', colors=cols[i-1], linewidth = 0.5)
    plt.legend(frameon=True, loc='lower center', ncol=1, fontsize = 5, bbox_to_anchor = (1.1, 0.5))
    plt.title('Residual plot of %s'%term, fontsize = 7)
    plt.xlabel('Data point', fontsize = 5)
    plt.ylabel('Residual', fontsize = 5)
    x_tick2 = np.arange(0, len(resd2), 10)
    plt.xticks(x_tick2, x_tick2)
    plt.grid()
    plt.show()
    
def sep_res_plot(a1, b1,indices2):
    typle_list = ['Oil', 'GC', 'Gas']
    title_list = ['Oil', 'GC', 'Gas']
    for term in range(3):
        ind1 = [True if typle_list[term] in i else False for i in indices2]
        oil_test = a1.values.ravel()[ind1]
        oil_pred = b1[ind1]
        new_resplot(oil_test, oil_pred, title_list[term])


# ### RF

# In[121]:


sep_res_plot(y_test3, _, test_ind)


# ### SVR

# In[124]:


sep_res_plot(y_test3, z, test_ind)


# ### LGBM

# In[198]:


sep_res_plot(y_test3, z2, test_ind)


# In[ ]:




