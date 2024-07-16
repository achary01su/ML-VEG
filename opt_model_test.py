import numpy as np
from data_loader import load_system_data, load_all_data

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_validate
#
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

# Load ML models
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

def set_models():
    
    """
    Sets up a dictionary of models for different chemical systems.

    Returns
    -------
    dict
        A dictionary containing models for each chemical system.
    """
    
    pipe3 = Pipeline([('preprocessor', PolynomialFeatures(degree=3)),('estimator', LinearRegression())])
    
    model_dict = {'benzene' :  {'LR' : LinearRegression(), 
                                'PR3': pipe3,
                                'KRR': KernelRidge(kernel='rbf', alpha = 1e-6, gamma = 1e-5),
                                'MLP': MLPRegressor(max_iter=10000, early_stopping=True, activation = 'relu',
                                                    hidden_layer_sizes = (80,60), alpha = 0.1, epsilon = 1e-5),
                                'ETR': ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=3, 
                                                           min_samples_split=8, n_estimators=200)
                               },
                  
                 'phenol' : { 'LR' : LinearRegression(), 
                              'PR3': pipe3,
                              'KRR': KernelRidge(kernel='rbf', alpha = 1e-6, gamma = 1e-5),
                              'MLP': MLPRegressor(max_iter=10000, early_stopping=True, activation = 'relu',
                                                  hidden_layer_sizes = (80,60), alpha = 1e-3, epsilon = 1e-5),
                              'ETR': ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=3, 
                                                           min_samples_split=8, n_estimators=200)
                            },
                  
                 'phenolate' : {'LR' : LinearRegression(), 
                                'PR3': pipe3,
                                'KRR': KernelRidge(kernel='rbf', alpha = 1e-6, gamma = 1e-5),
                                'MLP': MLPRegressor(max_iter=10000, early_stopping=True, activation = 'relu',
                                                    hidden_layer_sizes = (80,60), alpha = 1e-2, epsilon = 1e-5),
                                'ETR': ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=3, 
                                                           min_samples_split=8, n_estimators=200)
                               },
                  
                  'indole'   : {'LR'  : LinearRegression(), 
                                'PR3' : pipe3,
                                'KRR' : KernelRidge(kernel='rbf', alpha = 1e-6, gamma = 1e-5),
                                'MLP' : MLPRegressor(max_iter=10000, early_stopping=True, activation = 'relu',
                                                     hidden_layer_sizes = (80,60), alpha = 0.1, epsilon = 1e-4),
                                'ETR': ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=3, 
                                                           min_samples_split=8, n_estimators=200)
                               },
                  
                  'lumiflavin':{'LR' : LinearRegression(), 
                                'PR3': pipe3,
                                'KRR': KernelRidge(kernel='rbf', alpha = 1e-6, gamma = 1e-6),
                                'MLP': MLPRegressor(max_iter=10000, early_stopping=True, activation = 'relu',
                                                    hidden_layer_sizes = (80,60), alpha = 1.0, epsilon = 1e-4),
                                'ETR': ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=3, 
                                                           min_samples_split=8, n_estimators=200)
                               },
                 
                 'All' :       {'LR' : LinearRegression(), 
                                'PR3': pipe3,
                                'KRR': KernelRidge(kernel='rbf', alpha = 1e-5, gamma = 1e-2),
                                'MLP': MLPRegressor(max_iter=10000, early_stopping=True, activation = 'relu',
                                                    hidden_layer_sizes = (80,60), alpha = 0.1, epsilon = 1e-5),
                                'ETR': ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=3, 
                                                           min_samples_split=8, n_estimators=200)
                               }
                    
                 }
    
    return model_dict

def test_models(model_dict, df_id='0.0', validation=0.2, method='HF'):
    
    """
    Tests and evaluates models on different systems using cross-validation.

    Parameters
    ----------
    model_dict : dict
        A dictionary containing models for each system.
    df_id : str, optional
        Cutoff value for the data, either '0.0' or '7.5', by default '0.0'.
    validation : float, optional
        Proportion of the data to be used as validation set, by default 0.2.
    method : str, optional
        Method used for generating the feature lists, either 'HF' or 'EMP', by default 'HF'.

    Returns
    -------
    tuple
        A tuple containing scores dictionary and best models dictionary.
    """
    
    systems = model_dict.keys()
    score_list = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    
    scores_dict = {key : [ ] for key in systems}
    best_models = { }
    
    #print(list(systems))
    for system in list(systems):
        if system == "All":
            X, X_valid, y, y_valid = load_all_data(df_num=df_id, validation=validation, method=method)
        else:
            X, X_valid, y, y_valid = load_system_data(system, df_num=df_id, validation=validation, method=method)
    
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        if (validation != 0.0):
            scaled_X_valid = scaler.transform(X_valid)
            
        modelnames = model_dict[system].keys()
        
        for model in modelnames:
            
            m = model_dict[system][model]
            scores = cross_validate(m, scaled_X, y, cv=5, n_jobs=-1, return_train_score=True, scoring=score_list, return_estimator=True)
            
            #scores_dict[system] = scores
            #print(system, m, scores.keys())
            
            estm = scores['estimator'][0] 
            train_MAE_mean = -1.0*np.mean(scores['train_neg_mean_absolute_error'])
            test_MAE_mean = -1.0*np.mean(scores['test_neg_mean_absolute_error'])
            train_MSE_mean = -1.0*np.mean(scores['train_neg_mean_squared_error'])
            test_MSE_mean = -1.0*np.mean(scores['test_neg_mean_squared_error'])
            train_r2_mean = np.mean(scores['train_r2'])
            test_r2_mean = np.mean(scores['test_r2'])
            
            scores_dict[system].append([estm, train_MAE_mean, test_MAE_mean, train_MSE_mean, test_MSE_mean,
                                      train_r2_mean, test_r2_mean])
        
        best_m = sorted(scores_dict[system], key=lambda x:x[2])[0][0]
        best_models[system] = best_m
            
    return (scores_dict, best_models)
            

