# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features=CTG_features.drop(columns=extra_feature)
    CTG_features=CTG_features.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    c_ctg = copy.copy(CTG_features.to_dict())
    new_dict = {}
    for key1, value1 in c_ctg.items():
        new_dict.update({key1: [value2 for key2, value2 in value1.items() if (type(value2)==int or type(value2)==float)]})
    c_ctg = new_dict
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features = CTG_features.drop(columns=extra_feature)
    CTG_features = CTG_features.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    for key1 in CTG_features.columns:
        value1 = list(CTG_features[key1].dropna().values)
        c_cdf[key1] = CTG_features[key1].fillna(pd.Series(np.random.choice(value1, size=len(value1))))

    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for col in c_feat:
        small_dict = {'min': 0, '25%': 0, '50%': 0, '75%': 0, 'max': 0}
        Q = pd.DataFrame(c_feat[col])
        for key in small_dict:
            small_dict[key] = Q.describe().loc[key].values[0]
        d_summary[col] = small_dict
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for feat in d_summary:
        LF, UF = CalculateBoundaries(d_summary[feat])
        temp = c_feat[feat]
        c_no_outlier[feat] = temp[(temp <= UF) & (temp >= LF)]
    return pd.DataFrame(c_no_outlier)

def CalculateBoundaries(feat_summary):
    IQR = feat_summary['75%'] - feat_summary['25%']
    LF = feat_summary['25%'] - 1.5 * IQR
    UF = feat_summary['75%'] + 1.5 * IQR
    return LF, UF



def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    temp = c_cdf.loc[:, feature].to_numpy()
    filt_feature = temp[temp < thresh]
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    if mode == 'standard':
        # Standardization
        nsd_res = (CTG_features - CTG_features.mean()) / (CTG_features.std())
    elif mode == 'MinMax':
        # MinMax scaling / Normalization
        nsd_res = (CTG_features - CTG_features.min()) / (CTG_features.max() - CTG_features.min())
    elif mode == 'mean':
        # mean normalization
        nsd_res = (CTG_features - CTG_features.mean()) / (CTG_features.max() - CTG_features.min())
    else:
        nsd_res = CTG_features

    if flag:
        plt.title(mode, fontsize=20)
        CTG_features.loc[:, x].hist(bins=50, figsize=(15, 10), label='Original Data')
        nsd_res.loc[:, x].hist(bins=50, figsize=(15, 10), label='Scaled Data')
        plt.xlabel(x, fontsize=20)
        plt.ylabel('count', fontsize=20)
        plt.legend(loc='upper right', fontsize=20)
        plt.show()
        plt.title(mode, fontsize=30)
        CTG_features.loc[:, y].hist(bins=50, figsize=(15, 10), label='Original Data')
        nsd_res.loc[:, y]

    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
