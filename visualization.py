import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def plot_parity(y_predict, y_valid, figname):
    
    """
    Plot a parity plot between predicted and actual values.

    Parameters
    ----------
    y_predict : array-like
        Predicted values.
    y_valid : array-like
        Actual values.
    figname : str
        Name of the figure.

    Returns
    -------
    None
    """
    
    m_short = {'LinearRegression' : 'LR', 'Pipeline' : 'PR', 'KernelRidge' : 'KRR',
              'MLPRegressor' : 'MLP', 'ExtraTreesRegressor' : 'ETR', 'RandomForestRegressor' : 'RFR'}
    
    mae = mean_absolute_error(y_valid, y_predict)
    rmse = np.sqrt(mean_squared_error(y_valid, y_predict))
    r2 = r2_score(y_valid, y_predict)
    
    sys = figname.split('.sav')[0].split('_')[0].capitalize()
    #if sys == 'Lumifl':
    #    sys += 'avin'

    mod = m_short[figname.split('.sav')[0].split('_')[-1]]
    
    fig, ax = plt.subplots(1, figsize=(4.04,3.54))
    
    ax.set_xlabel(r"DFT $\Delta$E (eV)", fontsize=16)
    ax.set_ylabel(r"ML $\Delta$E (eV)", fontsize=16)
    ax.set_title(f"{sys} - {mod}", loc='center', fontsize=16, pad=5.0)
    
    xmin = np.minimum(y_predict.min(), y_valid.min()) - 0.05
    xmax = np.maximum(y_predict.max(), y_valid.max()) + 0.05
    
    ax.set_xlim(xmin-0.2,xmax+0.2)
    ax.set_ylim(xmin-0.2,xmax+0.2)
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Calculate the point density
    xy = np.vstack([y_predict,y_valid])
    z = gaussian_kde(xy)(xy)
    
    sc = ax.scatter(y_valid, y_predict, c=z, s=20, cmap='viridis')
    ax.plot([xmin, xmax],[xmin, xmax])
    
    cbar = fig.colorbar(sc)
    cbar.set_label("KDE Density") 
    
    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    plt.figtext(0.15, 0.82, f'MAE: {mae:5.3f}', fontsize=12)
    plt.figtext(0.15, 0.77, f'RMSE: {rmse:5.3f}', fontsize=12)
    plt.figtext(0.15, 0.72,  f'R$^2$: {r2:5.3f}', fontsize=12)
    
    name = figname.split(".sav")[0] + '_parity.png'
    plt.savefig(name, bbox_inches='tight', dpi=450, pad_inches=0.03)
    
    #with open(name, "w") as o:
    #    for ii in range(len(y_predict)):
    #        o.write(f"{y_valid.iloc[ii]}\t{y_predict[ii]}\n")
    #plt.show()
    plt.close()
    
def plot_learning_curve(est, X_train, y_train, figname):
    
    """
    Plot learning curve for a given estimator.

    Parameters
    ----------
    est : estimator
        The estimator to plot the learning curve for.
    X_train : array-like
        Training input samples.
    y_train : array-like
        Target values.
    figname : str
        Name of the figure.

    Returns
    -------
    None
    """
    
    m_short = {'LinearRegression' : 'LR', 'Pipeline' : 'PR', 'KernelRidge' : 'KRR',
              'MLPRegressor' : 'MLP', 'ExtraTreesRegressor' : 'ETR', 'RandomForestRegressor' : 'RFR'}
    
    sys = figname.split('.sav')[0].split('_')[0].capitalize()
    #if sys == 'Lumifl':
    #    sys += 'avin'

    mod = m_short[figname.split('.sav')[0].split('_')[-1]]
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    fig, ax = plt.subplots(1, figsize=(4.04,3.54))
    
    ax.set_xlabel("Training data size", fontsize=14)
    ax.set_ylabel("MAE (eV)", fontsize=14)
    ax.set_title(f"{sys} - {mod}", loc='center', fontsize=16, pad=5.0)
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    #Calculate the train sizes, scores and test scores using the learning_curve function
    train_sizes, train_scores, test_scores = learning_curve(est, X_train, y_train, scoring = 'neg_mean_absolute_error',
                                             cv=5, train_sizes=train_sizes)
    
    # Calculate training and test mean and std
    train_mean = np.mean(-train_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    test_mean = np.mean(-test_scores, axis=1)
    test_std = np.std(-test_scores, axis=1)
    
    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # Plot the learning curve
    ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training')
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation')
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    
    name = figname.split(".sav")[0] + '_lc1.png'
    plt.savefig(name, bbox_inches='tight', dpi=450, pad_inches=0.02)
    
    #plt.show()
    plt.close()
    

def make_score_heatmap(test_dict, prefix):
    
    """
    Make a heatmap of scores for different models and systems.

    Parameters
    ----------
    test_dict : dict
        Dictionary containing test scores.
    prefix : str
        Prefix for the output
        
    Returns
    -------
    None
    """
    
    keys = [key for key in test_dict.keys()]
    models = [type(test_dict[key][i][0]).__name__ for key in test_dict.keys() for i in range(len(test_dict.keys())-1)]
    results = [test_dict[key][i][1:] for key in test_dict.keys() for i in range(len(test_dict.keys())-2)]
    #print(keys)
    #print(models)
    #print(results)

    mae_dict = {k : {} for k in keys}
    mse_dict = {k : {} for k in keys}


    for key in keys:
        for i, model in enumerate(models[:5]):
            mae_dict[key][model] = test_dict[key][i][2]
            mse_dict[key][model] = test_dict[key][i][4]
            #print(i, test_dict[key][i][1:])
        
    #mae_dict
    mae_df = pd.DataFrame(mae_dict)
    mse_df = pd.DataFrame(mse_dict)
    
    # Update xlab, ylab, and ax after adding new systems or models
    xlab = ['benzene', 'phenol','phenolate','indole', 'lumiflavin', 'All']
    ylab = ['LR', 'PR', 'KRR', 'MLP', 'ETR']
    ax = sns.heatmap(mae_df.iloc[:,:6], xticklabels=xlab, yticklabels=ylab, annot=True)
    sns.set_theme(font='sans-serif', style="ticks")

    figure = ax.get_figure()
    fname = prefix + "_mae_heat.png"
    figure.savefig(fname, bbox_inches='tight', dpi=450, pad_inches=0.03)
    
    plt.clf()
    ax2 = sns.heatmap(np.sqrt(mse_df.iloc[:,:6]), xticklabels=xlab, yticklabels=ylab, annot=True)
    sns.set_theme(font='sans-serif', style="ticks")
    
    figure2 = ax2.get_figure()
    fname2 = prefix + "_rmse_heat.png"
    figure2.savefig(fname2, bbox_inches='tight', dpi=450, pad_inches=0.03)

    plt.close()
