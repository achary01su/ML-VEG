import numpy as np
import itertools
from data_loader import load_system_data, load_all_data

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import  KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

# Load ML models
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor


def model_dict():
    
    """
    Creates a dictionary of machine learning models with their corresponding parameter grids.

    Returns
    -------
    dict
        A dictionary where keys are model names and values are dictionaries with
        'model' and 'param_grid' keys.
    """
    
    pipe2 = Pipeline([('preprocessor', PolynomialFeatures(degree=2)),
                    ('estimator', LinearRegression())])
    pipe3 = Pipeline([('preprocessor', PolynomialFeatures(degree=3)),
                    ('estimator', LinearRegression())])
    pipe4 = Pipeline([('preprocessor', PolynomialFeatures(degree=4)),
                    ('estimator', LinearRegression())])
    
    dict_ml = {'LR' :  { 'model' : LinearRegression(), 'param_grid' : {}},
               'KRR' : { 'model' : KernelRidge(), 'param_grid' : {"alpha" : np.logspace(-6, 2, 9), "gamma" : np.logspace(-14, 1, 16), "kernel" : ['rbf']}},
               'MLP' : { 'model' : MLPRegressor(max_iter = 1000), "param_grid" : {'hidden_layer_sizes' : [x for x in itertools.product((40,60,70),repeat=2)], 'activation' : ['relu'], 
               #                                                                                         'alpha' : np.logspace(-4, 1, 6), 'epsilon' : np.logspace(-5, -1, 5)}},
               'PR2' : { 'model' : pipe2, 'param_grid' : {}},
               'PR3' : { 'model' : pipe3, 'param_grid' : {}},
               'PR4' : { 'model' : pipe4, 'param_grid' : {}},
               'ETR' : { 'model' : ExtraTreesRegressor(), 'param_grid' : {'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21),
                                                                          'min_samples_leaf': range(1, 21), "n_estimators" : np.arange(100,1000,100)}},
              }
    
    return dict_ml

def nestedCV2(X, y, X_valid, y_valid, model, param_grid):
    
    """
    Performs nested cross-validation to evaluate the performance of a machine learning model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    X_valid : array-like, shape (n_samples, n_features)
        Validation data.
    y_valid : array-like, shape (n_samples,)
        Validation target values.
    model : estimator object
        The machine learning model to evaluate.
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.

    Returns
    -------
    dict
        A dictionary containing performance metrics for the model.
    """
    
    results = {'MAE_train' : [],
              'MAE_test' : [],
              'MAE_valid' : [],
              'r2_train' : [],
              'r2_test' : [],
              'r2_valid' : [],
              'MSE_train': [],
              'MSE_test' : [],
              'MSE_valid' : []}
    
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
    n_loop = 1
    
    for train_o, test_o in cv_outer.split(X):
        
        print(f"Loop number {n_loop} of 5")
        n_loop += 1
        
        X_train, X_test = X[train_o, :], X[test_o, :]
        y_train, y_test = y.iloc[train_o], y.iloc[test_o]
        
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        
        GridSearch = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', 
                                  cv = cv_inner, refit = True, n_jobs=-1)
        
        GridSearch.fit(X_train, y_train)
        
        try:
            #print("best params: ", GridSearch.get_params().keys())
            print("best estimator: ", GridSearch.best_estimator_)
            #print("best estimator: ", GridSearch.best_score_)
        except:
            print("couldn't get parameters")
        
        y_train_pred = GridSearch.predict(X_train)
        y_test_pred = GridSearch.predict(X_test)
        y_valid_pred = GridSearch.predict(X_valid)
        
        results['MAE_train'].append(mean_absolute_error(y_train_pred, y_train))
        results['MAE_test'].append(mean_absolute_error(y_test_pred, y_test))
        results['MAE_valid'].append(mean_absolute_error(y_valid_pred, y_valid))
        
        results['MSE_train'].append(mean_squared_error(y_train_pred, y_train))
        results['MSE_test'].append(mean_squared_error(y_test_pred, y_test))
        results['MSE_valid'].append(mean_squared_error(y_valid_pred, y_valid))
        
        results['r2_train'].append(r2_score(y_train_pred, y_train))
        results['r2_test'].append(r2_score(y_test_pred, y_test))
        results['r2_valid'].append(r2_score(y_valid_pred, y_valid))
        
    results['MAE_train_mean'] = np.mean(np.array(results['MAE_train']))
    results['MAE_test_mean'] = np.mean(np.array(results['MAE_test']))
    results['MAE_valid_mean'] = np.mean(np.array(results['MAE_valid']))
    
    results['MSE_train_mean'] = np.mean(np.array(results['MSE_train']))
    results['MSE_test_mean'] = np.mean(np.array(results['MSE_test']))
    results['MSE_valid_mean'] = np.mean(np.array(results['MSE_valid']))
    
    results['r2_train_mean'] = np.mean(np.array(results['r2_train']))
    results['r2_test_mean'] = np.mean(np.array(results['r2_test']))
    results['r2_valid_mean'] = np.mean(np.array(results['r2_valid']))
    
    results['MAE_train_std'] = np.std(np.array(results['MAE_train']))
    results['MAE_test_std'] = np.std(np.array(results['MAE_test']))
    results['MAE_valid_std'] = np.std(np.array(results['MAE_valid']))
    
    results['MSE_train_std'] = np.std(np.array(results['MSE_train']))
    results['MSE_test_std'] = np.std(np.array(results['MSE_test']))
    results['MSE_valid_std'] = np.std(np.array(results['MSE_valid']))
    
    results['r2_train_std'] = np.std(np.array(results['r2_train']))
    results['r2_test_std'] = np.std(np.array(results['r2_test']))
    results['r2_valid_std'] = np.std(np.array(results['r2_valid']))
    
    return results

def optimize_hyperparameters(system, df_id='0.0', validation=0.1, method='HF'):
       
    """
    Optimizes hyperparameters for various machine learning models using nested cross-validation.

    Parameters
    ----------
    system : str
        The chemical system for which data is required.
        Valid options are specific system names or 'All' for all systems.
    df_id : str, optional
        Cutoff value for the data, either '0.0' or '7.5', by default '0.0'.
    validation : float, optional
        Proportion of the data to be used as validation set, by default 0.1.
    method : str, optional
        Method used for generating the feature lists, either 'HF' or 'EMP', by default 'HF'.
    """
    
    if system == "All":
        X, X_valid, y, y_valid = load_all_data(df_num=df_id, validation=validation, metod=method)
    else:
        X, X_valid, y, y_valid = load_system_data(system, df_num=df_id, validation=validation, method=method)
    
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    scaled_X_valid = scaler.transform(X_valid)
    
    models = model_dict()
    
    print(f"Results for: {system} {df_id} {method}")
    for model in models.keys():

        name = models[model]['model']
        param_grid = models[model]['param_grid']

        #scores = nestedCV(scaled_X, y, name, param_grid)
        scores2 = nestedCV2(scaled_X, y, scaled_X_valid, y_valid, name, param_grid)

        #print(name, -scores)
        #print(name)
        fname = str(system) + '_' + str(method) + '_' + str(df_id)  + '_scores.dat'
        with open(fname, 'a') as f:
            f.write(f"\n{model}\n")
            f.write(f"Train MAE: {scores2['MAE_train_mean']}+-{scores2['MAE_train_std']}\n")
            f.write(f"Test MAE: {scores2['MAE_test_mean']}+-{scores2['MAE_test_std']}\n")
            f.write(f"Valid MAE: {scores2['MAE_valid_mean']}+-{scores2['MAE_valid_std']}\n")

            f.write(f"Train MSE: {scores2['MSE_train_mean']}+-{scores2['MSE_train_std']}\n")
            f.write(f"Test MSE: {scores2['MSE_test_mean']}+-{scores2['MSE_test_std']}\n")
            f.write(f"Valid MSE: {scores2['MSE_valid_mean']}+-{scores2['MSE_valid_std']}\n")

            f.write(f"Train R2: {scores2['r2_train_mean']}+-{scores2['r2_train_std']}\n")
            f.write(f"Test R2: {scores2['r2_test_mean']}+-{scores2['r2_test_std']}\n")
            f.write(f"Valid R2: {scores2['r2_valid_mean']}+-{scores2['r2_valid_std']}\n")

    print("---------------------------------------")

if __name__ == "__main__" : 
    
    systems = ['benzene', 'phenol', 'phenolate', 'indole', 'lumiflavin']
    methods = ['HF', 'EMP']
    df_list = ['0.0', '7.5']
    valid = 0.2 
    
    for system in systems:
        for met in methods:
            for df in df_list:
                optimize_hyperparameters(system, df_id = df, validation = valid, method = met)
