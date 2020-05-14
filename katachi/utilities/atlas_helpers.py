# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:43:54 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Module with functions used for testing and analysis of secondary
            channel atlas generation by multivariate-multivariable regression.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import spearmanr


#------------------------------------------------------------------------------

# SCORING FUNCTIONS FOR REPORTING SPEARMAN RANK CORRELATIONS

def spearman_r_score(y, y_pred):
    """Compute mean spearman rank correlation r score across dimensions of y.
    """
    scores = [spearmanr(y[:,dim], y_pred[:,dim])[0]
              for dim in range(y.shape[1])]
    return np.mean(scores)

def spearman_p_bonf(y, y_pred):
    """Compute mean spearman rank corrrelation p value across dimensions of y
    with Bonferroni multiple testing correction.
    """
    scores = [spearmanr(y[:,dim]*y.shape[0], y_pred[:,dim])[1]
              for dim in range(y.shape[1])]
    return np.mean(scores)


#------------------------------------------------------------------------------

# FUNCTION TO REPORT SCORES FROM CROSS VALIDATION

def report_cv_scores(scores):
    """Print structured report of mean and standard deviation of `scores` dict
    produced by `sklearn.model_selection.cross_validate`.
    """

    print "\nReporting cross-validation scores:"

    # Prep sorted keypairs
    keypairs = []
    for key in sorted(scores.keys()):
        if key.startswith("train"):
            keypairs.append((key, "test"+key[5:]))

    # Report
    for keypair in keypairs:
        print "\n  "+keypair[0][6:]
        print "    Train: {:6.3f}  +/- {:3.3f}".format(
                                                  np.mean(scores[keypair[0]]),
                                                  np.std(scores[keypair[0]]))
        print "    Test:  {:6.3f}  +/- {:3.3f}".format(
                                                  np.mean(scores[keypair[1]]),
                                                  np.std(scores[keypair[1]]))


#------------------------------------------------------------------------------

# FUNCTION TO REPORT ON THE PERFORMANCE OF A REGRESSOR

def visualize_regression(estimator, estimator_name,
                         X_train, X_test, y_train, y_test,
                         plot_residuals=True, do_residuals_pca=True,
                         plot_dim_scores=True):
    """Create and show residuals plot and per-dimension spearman r score plot
    for a given regressor and train-test splitted dataset.

    WARNING: This does not work as intended if the regressor performs any
    preprocessing on y!

    Parameters
    ----------
    estimator : sklearn estimator object
        A scikit-learn regressor.
    estimator_name : string
        The name of the regressor (used as plot title).
    X_train, X_test, y_train, y_test : ndarrays of shape (samples, features)
        A train-test splitted dataset.
    plot_residuals : bool, optional, default True
        If True, the residuals plot is generated.
    do_residuals_pca : bool, optional, default True
        If True, the residuals plot shows the first two dimensions of a PCA on
        the training data. Otherwise, the first two dimensions of the feature
        array are shown.
    plot_dim_scores : bool, optional, default True
        If True, a plot showing spearman rank correlation r scores for each
        dimension of the target data (y) is generated.
    """

    print "\nVisualizing regression results..."

    #--------------------------------------------------------------------------

    ### Create predictions

    # Train
    print "  Fitting model..."
    estimator.fit(X_train, y_train)

    # Predict on training data
    print "  Predicting on training set..."
    y_train_pred = estimator.predict(X_train)

    # Predict on test data
    print "  Predicting on test set..."
    y_test_pred = estimator.predict(X_test)


    #--------------------------------------------------------------------------

    ### Plot residuals (PCA-reduced)

    if plot_residuals:

        # Perform PCA on training samples
        if do_residuals_pca:
            pca = sklearn.decomposition.PCA(n_components=2)
            y_tr_rdy   = pca.fit_transform(y_train)
            y_tr_p_rdy = pca.transform(y_train_pred)
            y_te_rdy   = pca.transform(y_test)
            y_te_p_rdy = pca.transform(y_test_pred)
        else:
            y_tr_rdy   = y_train
            y_tr_p_rdy = y_train_pred
            y_te_rdy   = y_test
            y_te_p_rdy = y_test_pred


        # Prep fig
        fig,ax = plt.subplots(1, 2, figsize=(12,5))

        # Plot prediction on training set
        for sample in np.arange(y_tr_rdy.shape[0]):
            ax[0].plot([y_tr_rdy[sample,0], y_tr_p_rdy[sample,0]],
                       [y_tr_rdy[sample,1], y_tr_p_rdy[sample,1]],
                       'k', zorder=0, alpha=0.25)
        ax[0].scatter(y_tr_rdy[:,0], y_tr_rdy[:,1],
                      c='cyan', edgecolor='', s=5, label="Ground Truth")
        ax[0].scatter(y_tr_p_rdy[:,0], y_tr_p_rdy[:,1],
                      c='darkblue', edgecolor='', s=5,  label="Prediction")

        # Axis cosmetics
        ax[0].legend()
        ax[0].set_title("Prediction on Training Set")
        if do_residuals_pca:
            ax[0].set_xlabel("target PC 0")
            ax[0].set_ylabel("target PC 1")
        else:
            ax[0].set_xlabel("target dim 0")
            ax[0].set_ylabel("target dim 1")

        # Plot prediction on test set
        for sample in np.arange(y_te_rdy.shape[0]):
            ax[1].plot([y_te_rdy[sample,0], y_te_p_rdy[sample,0]],
                       [y_te_rdy[sample,1], y_te_p_rdy[sample,1]],
                       'k', zorder=0, alpha=0.25)
        ax[1].scatter(y_te_rdy[:,0], y_te_rdy[:,1],
                      c='cyan', edgecolor='', s=5, label="Ground Truth")
        ax[1].scatter(y_te_p_rdy[:,0], y_te_p_rdy[:,1],
                      c='darkblue', edgecolor='', s=5,  label="Prediction")

        # Axis cosmetics
        ax[1].legend()
        ax[1].set_title("Prediction on Test Set")
        if do_residuals_pca:
            ax[1].set_xlabel("target PC 0")
        else:
            ax[1].set_xlabel("target dim 0")
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())

        # Finish up
        plt.suptitle(estimator_name, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


    #--------------------------------------------------------------------------

    ### Plot scores for each output dimension

    if plot_dim_scores:

        # Score
        spear_train = [spearmanr(y_train[:,dim], y_train_pred[:,dim])[0]
                       for dim in range(y_train.shape[1])]
        spear_test  = [spearmanr(y_test[:,dim], y_test_pred[:,dim])[0]
                       for dim in range(y_test.shape[1])]

        # Prep
        fig,ax = plt.subplots(1, 1, figsize=(12,3))

        # Plot scores for training set
        ax.plot(spear_train, 'k.-', alpha=0.25, label='train')

        # Plot scores for test set
        ax.plot(spear_test, 'k.-', label='test')

        # Cosmetics
        ax.legend(loc=4)
        ax.set_xticks(np.arange(0, y_train.shape[1], 1))
        ax.set_xlim([-0.2, y_train.shape[1]-0.8])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel('target dimensions')
        ax.set_ylabel('spearman r-score')
        ax.set_title('Target Dim Predictability',
                     fontsize=10)

        # Done
        plt.show()


#------------------------------------------------------------------------------



