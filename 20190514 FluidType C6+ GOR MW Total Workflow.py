#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, accuracy_score, recall_score, confusion_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd
import copy

import seaborn as sns
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches


# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
set_matplotlib_formats('retina')
plt.rcParams['figure.dpi'] = 250
plt.rcParams.update({'font.size': 5})
sns.set_style('dark', {'legend.frameon':True})
np.random.seed(42)


## Function of feature engineering
def feat_eng(total_x):
    total_x = total_x.div(total_x.sum(axis = 1), axis = 0)*100
    total_x['Wh'] = total_x[['C2_mol', 'C3_mol', 'C4_mol', 'C5_mol']].sum(axis = 1)*100 / total_x[['C1_mol', 'C2_mol', 'C3_mol', 'C4_mol', 'C5_mol']].sum(axis = 1)
    total_x['Bh'] = (total_x['C1_mol'] + total_x['C2_mol']) / (total_x['C3_mol'] + total_x['C4_mol'] + total_x['C5_mol'])
    total_x['Ch'] = (total_x['C4_mol'] + total_x['C5_mol']) / total_x['C3_mol']
    return total_x

## Function for clustering
def myclustering(total_x_r, total_y):
    clusters = 2
    gmm = make_pipeline(StandardScaler(), GaussianMixture(n_components=clusters, random_state=0))
#     print((total_x_r.C1_mol.shape), (total_y['C6plus_mol'].values.reshape(-1,1).shape))
    _c_data = np.column_stack((total_x_r.C1_mol, total_y['C6plus_mol'].values.reshape(-1,1)))
    labels = gmm.fit(_c_data).predict(_c_data)
    return labels


# In[4]:


def get1stclassData(X_train, y_train, X_val, y_val, lab):
    x_tr0, y_tr0, x_va0, y_va0 = X_train[y_train.Labels == lab], y_train[y_train.Labels == lab], X_val[y_val.Labels == lab], y_val[y_val.Labels == lab]
    return x_tr0, y_tr0, x_va0, y_va0

def get1stclassData_2cluster(X_train, y_train, X_val, y_val, lab):
    x_tr0, y_tr0, x_va0, y_va0 = X_train[y_train.New_label == lab], y_train[y_train.New_label == lab], X_val[y_val.New_label == lab], y_val[y_val.New_label == lab]
    return x_tr0, y_tr0, x_va0, y_va0

def joinC1andC6(X_train, y_train):
    return pd.concat((X_train.C1_mol, np.log1p(y_train.C6plus_mol)), axis = 1)

def dataFluidType(fn, clf = None, gor = None):
    training_features = pd.read_csv(fn)

    total_x = training_features[['C1_mol', 'C2_mol', 'C3_mol', 'C4_mol', 'C5_mol']]
    total_x = total_x.fillna(0)

    total_y = copy.copy(training_features[['C6plus_mol', 'GOR_scf_stb']])

    # Feature engineering
    total_x_r = feat_eng(total_x)

    # The first clustering
    total_y['Labels'] = myclustering(total_x_r, total_y).ravel().astype(int)
    # Arbituary standard
    total_y.loc[total_y.C6plus_mol<18,'Labels'] = 0

    # The second clustering
    total_y['Labels2'] = np.zeros((total_y.shape[0], 1)).astype(int)
    total_x_r2 = total_x_r[total_y['Labels'] == 1]
    total_y2 = copy.copy(total_y[total_y['Labels'] == 1])
    total_y.loc[total_y.Labels == 1, 'Labels2'] = myclustering(total_x_r2, total_y2).ravel().astype(int)+1

    # Renaming the labels
    total_y['Type'] = np.zeros((total_y.shape[0], 1)).astype(int)
    total_y.loc[total_y.Labels2==0, 'Type'] = 'GC'
    total_y.loc[total_y.Labels2==1, 'Type'] = 'BO1'
    total_y.loc[total_y.Labels2==2, 'Type'] = 'BO2'

    total_y['Type2'] = np.zeros((total_y.shape[0], 1)).astype(int)
    total_y.loc[total_y.Labels==0, 'Type2'] = 'GC'
    total_y.loc[total_y.Labels==1, 'Type2'] = 'Oil'
    
    if clf != None:
        ## Remove outliers
        ind = (np.log1p(total_x_r.C1_mol) > 1).values
        total_x_r = total_x_r[ind]
        total_y = total_y[ind]
        
    if gor != None:
        ind2 = (~total_y.GOR_scf_stb.isnull())&(total_y.GOR_scf_stb > 0)
        total_y = total_y[ind2]
        total_x_r = total_x_r[ind2]
        
    ## Merge GC and BO2 to on cluster
    total_y['New_label'] = np.zeros((total_y.shape[0], 1)).astype(int)
    total_y.loc[(total_y.Labels2==0)|(total_y.Labels2==2), 'New_label'] = 1
    
    ## Split
    X_train, X_val, y_train, y_val = train_test_split(total_x_r, total_y, test_size = 0.2, random_state = 6265)
    X_train, X_val = np.log1p(X_train), np.log1p(X_val)
    
    return X_train, y_train, X_val, y_val

def dataMWpred(fn):
    training_features = pd.read_csv(fn)

    total_x = training_features[['C1_mol', 'C2_mol', 'C3_mol', 'C4_mol', 'C5_mol']]
    total_x = total_x.fillna(0)
    total_x_r = feat_eng(total_x)
    total_x_r2 = pd.concat((total_x_r, training_features.C6plus_mol), axis=1)

    log_C6 = np.log1p(training_features.C6plus_mol)
    log_mw = np.log1p(training_features.MW_C6plus_g_mole)

    # outliers removal
    ind2 = (log_mw>4.75) | ((log_mw>4.5)&(log_mw<4.75)&(log_C6<4))
    training_x2 = np.log1p(total_x_r2[ind2])
    training_y2 = log_mw[ind2]
#     # clustering
#     gmm = GaussianMixture(n_components=2, random_state=42)
#     lb = gmm.fit_predict(pd.concat((training_x2.C6plus_mol, training_y2), axis=1))

    training_y2 = pd.DataFrame(training_y2)
#     training_y2['Label'] = lb
    return training_x2, training_y2

def weighted_pre(rf_clf1, rf_clf2, rf_reg1, rf_reg2, X_test):
    X_test2 = copy.copy(X_test)
    
    # Classification only for GC and Oil
    pro1 = rf_clf1.predict_proba(X_test)
    X_test2['GC_probability, %'], X_test2['Oil_probability, %'] = pro1[:,0]*100, pro1[:,1]*100
    
    # Classification for C6+ prediction
    pro2 = rf_clf2.predict_proba(X_test)
    X_test2['C0_pro'], X_test2['C1_pro'] = pro2[:,0], pro2[:,1]

    X_test2['C6+_predictions, Mole%'] = np.expm1(X_test2['C0_pro']*rf_reg1.predict(X_test) + X_test2['C1_pro']*rf_reg2.predict(X_test))
    X_test3 = pd.concat((np.expm1(X_test[['C1_mol', 'C2_mol', 'C3_mol', 'C4_mol', 'C5_mol']]), X_test2[['GC_probability, %', 'Oil_probability, %', 'C6+_predictions, Mole%']]), axis = 1)
    return X_test3

def C6prediction(fn, X_test):
    X_train, y_train, X_val, y_val = dataFluidType(fn)

    ## The 1st classification, GC and Oil
    rf_clfbp_1st = {'criterion': 'entropy',
     'max_depth': 26,
     'max_features': 3,
     'n_estimators': 947}
    rf_clf_1st = RandomForestClassifier(**rf_clfbp_1st, random_state=42).fit((X_train), y_train.Type2)

    ## The 2nt classification, purely for predictions
    rf_clfbp_2nd = {'max_depth': 10, 'n_estimators': 558}
    rf_clf_2nd = RandomForestClassifier(**rf_clfbp_2nd, class_weight={0:10, 1:1}, random_state=42).fit((X_train), y_train.New_label)
    
    ## First non-linear regression, Cluster 0, RF, to predict C6+
    x_tr0, y_tr0, x_va0, y_va0 = get1stclassData_2cluster(X_train, y_train, X_val, y_val, 0)
    y_tr0_C6, y_va0_C6 = np.log1p(y_tr0.C6plus_mol), np.log1p(y_va0.C6plus_mol)
    
    rf_bf_reg1 = {'criterion': 'mae',
     'max_depth': 6,
     'max_features': 1,
     'n_estimators': 2760,
     'n_jobs': -1,
     'random_state': 42}
    rf_reg1 = RandomForestRegressor(**rf_bf_reg1).fit(x_tr0, y_tr0_C6)
    
    ## First non-linear regression, Cluster 1, RF, to predict C6+
    x_tr1, y_tr1, x_va1, y_va1 = get1stclassData_2cluster(X_train, y_train, X_val, y_val, 1)
    y_tr1_C6, y_va1_C6 = np.log1p(y_tr1.C6plus_mol), np.log1p(y_va1.C6plus_mol)
    
    rf_regbp2 = {'criterion': 'mae',
     'max_depth': 13,
     'max_features': 2,
     'n_estimators': 2260,
     'n_jobs': -1,
     'random_state': 42}
    rf_reg2 = RandomForestRegressor(**rf_regbp2).fit(x_tr1, y_tr1_C6)
    pred = weighted_pre(rf_clf_1st, rf_clf_2nd, rf_reg1, rf_reg2, X_test)
    
    return pred 

def MWprediction(fn, X_test):
    training_x, training_y = dataMWpred(fn)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(training_x, training_y, test_size = 0.2, random_state = 6265)

    #Separate MW and labels
#     y_train_label, y_val_label = y_train['Label'], y_val['Label']
    y_trainMW, y_valMW = y_train['MW_C6plus_g_mole'], y_val['MW_C6plus_g_mole']
    
    rf_regbp_raw = {'criterion': 'mae',
     'max_depth': 16,
     'max_features': 7,
     'n_estimators': 60,
     'n_jobs': -1,
     'random_state': 42}
    rf_reg1 = RandomForestRegressor(**rf_regbp_raw).fit(X_train, y_trainMW)
    rf1_pred = rf_reg1.predict(X_test)
    return rf1_pred

def GORprediction(fn, X_test):
    X_train, y_train, X_val, y_val = dataFluidType(fn, gor = True)
    
    gor_xtrain = joinC1andC6(X_train, y_train)
    gor_ytrain = np.log1p(y_train['GOR_scf_stb'])
    gor_xval = joinC1andC6(X_val, y_val)
    gor_yval = np.log1p(y_val['GOR_scf_stb'])
    
    g_bp = {'alpha': 0.06990939319645255, 'kernel': RBF(length_scale=38.7)}
    gpr = GaussianProcessRegressor(**g_bp).fit(gor_xtrain, gor_ytrain)
    return gpr.predict(X_test[['C1_mol', 'C6plus_mol']])
    
def main(X_test):
    fn = 'C:/YujianWu/Data/AMG/103111289_(original)_Extended dBase_processed_mol.csv'

    plot_features = pd.read_csv(fn)
    re_x = feat_eng(plot_features[['C1_mol', 'C2_mol', 'C3_mol', 'C4_mol', 'C5_mol']])
    
    if isinstance(X_test, list):
        X_test = pd.DataFrame(np.array(X_test).reshape(1,-1), columns=['C1_mol', 'C2_mol', 'C3_mol', 'C4_mol', 'C5_mol'])
        X_test = np.log1p(feat_eng(X_test))
    else:
        X_test = np.log1p(X_test)

    # predict C6+
    pred_m1 = C6prediction(fn, X_test)
    
    # predict MW
    X_test['C6plus_mol'] = np.log1p(pred_m1['C6+_predictions, Mole%'])
    pred_m1['MW_predictions, g/mole'] = np.expm1(MWprediction(fn, X_test))
    
    # predict GOR
#     display(X_test)
    pred_m1['GOR_predictions, SCF/STB'] = np.expm1(GORprediction(fn, X_test))
    
    print('The renormalized data is:')
    display(np.round(pred_m1.iloc[:,0:5], decimals=2))
    print('\n')
    print('The predicted values are:')
    display(np.round(pred_m1.iloc[:,5:], decimals=2))
    print()
##    # Figures
##    fs = 10
##    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(19,4.5))
##    
##    color_dict = {'GC_probability, %': sns.color_palette()[2], 'Oil_probability, %': sns.color_palette()[1]}
##    new_df = pred_m1[['GC_probability, %', 'Oil_probability, %']]
##    g = new_df.plot.barh(stacked=True, color=[color_dict.get(x, '#333333') for x in new_df.columns], ax=ax1)
##    ax1.set_xlabel('Probability', fontsize = fs)
##    ax1.set_title('Fluid Type Probability', fontsize = fs)
##    ax1.tick_params(axis = 'both', labelsize = fs)
##    ax1.legend(fontsize = fs)
##    ax1.grid()
##
##    sns.scatterplot(re_x.C1_mol, plot_features.C6plus_mol, label = 'Training', s = 10, linewidth = 0, ax=ax2)
##    sns.scatterplot(pred_m1.C1_mol, pred_m1['C6+_predictions, Mole%'], label = 'Test Prediction', linewidth = 0, ax=ax2)
##    ax2.set_title('C6plus Prediction', fontsize = fs)
##    ax2.set_xlabel('C1, Mole%', fontsize = fs)
##    ax2.set_ylabel('C6plus, Mole%', fontsize = fs)
##    ax2.tick_params(axis = 'both', labelsize = fs)
##    ax2.legend(fontsize = fs)
##    ax2.grid()
##    
##    sns.scatterplot(plot_features.C6plus_mol, np.log1p(plot_features.GOR_scf_stb), label = 'Training', s = 10, linewidth = 0, ax=ax3)
##    sns.scatterplot(pred_m1['C6+_predictions, Mole%'], np.log1p(pred_m1['GOR_predictions, SCF/STB']), label = 'Test Prediction', linewidth = 0, ax=ax3)
##    ax3.set_title('GOR Prediction', fontsize = fs)
##    ax3.set_xlabel('C6plus, Mole%', fontsize = fs)
##    ax3.set_ylabel('GOR, SCF/STB', fontsize = fs)
##    ax3.tick_params(axis = 'both', labelsize = fs)
##    ax3.legend(fontsize = fs)
##    ax3.grid()
##    
###     ax4.set_axis_off()
###     plt.grid()
##    plt.tight_layout()
##    plt.show()




