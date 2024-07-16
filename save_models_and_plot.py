import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_data, load_system_data
from opt_model_test import set_models, test_models
from visualization import plot_learning_curve, plot_parity, make_score_heatmap

def save_models(df_id='0.0', valid=0.2, method='EMP'):
    
    """
    Train and save machine learning models, and generate evaluation plots.

    This function sets up machine learning models, tests them, and then
    trains the best performing models on provided datasets. It generates
    and saves evaluation plots, such as score heatmaps, parity plots, and
    learning curves. 

    Parameters:
    -----------
    df_id : str, optional (default='0.0')
        Identifier for the dataset.
    valid : float, optional (default=0.2)
        Fraction of the data to be used for validation.
    method : str, optional (default='EMP')
        Method name or identifier used in the modeling process.

    Returns:
    --------
    None

    """

    models = set_models()
    results_dict, best_models = test_models(models, df_id=df_id, validation=valid, method=method)
    
    prefix = method + "_" + df_id
    #make_score_heatmap(results_dict, prefix)
    
    systems = results_dict.keys()
    
    scaler = StandardScaler()
    
    for system in systems:
        if system == "All":
            X, X_valid, y, y_valid = load_all_data(df_num=df_id, validation=valid, method=method)
            print(len(X), len(X_valid))
            model_names = [elem[0] for elem in results_dict[system]]
            print(model_names)
            for model_n in model_names:
                
                scaled_X_train = scaler.fit_transform(X)
                scaled_X_valid = scaler.transform(X_valid)
                
                model_n.fit(scaled_X_train, y)
                y_pred = model_n.predict(scaled_X_valid)
                y_train_pred = model_n.predict(scaled_X_train)
                
                print(f"Results for {system}, model {model_n}: RMSE: {np.sqrt(mean_squared_error(y_valid, y_pred))}")
                
                save_model_name = system + "_" + method + "_" + df_id + "_" + type(model_n).__name__ + '.sav'
                
                print(save_model_name)
                pickle.dump(model_n, open(save_model_name, 'wb'))
                plot_parity(y_pred, y_valid, save_model_name)
                plot_learning_curve(model_n, scaled_X_train, y, save_model_name)
                
        else:
            X, X_valid, y, y_valid = load_system_data(system, df_num=df_id, validation=valid, method=method)
            print(len(X), len(X_valid))
            model_n = best_models[system]
                
            scaled_X_train = scaler.fit_transform(X)
            scaled_X_valid = scaler.transform(X_valid)
                
            model_n.fit(scaled_X_train, y)
            y_pred = model_n.predict(scaled_X_valid)
                
            print(f"Results for {system}, model {model_n}: RMSE: {np.sqrt(mean_squared_error(y_valid, y_pred))}")
            
            save_model_name = system + "_" + method + "_" + df_id + "_" + type(model_n).__name__ + '.sav'
            print(save_model_name)
            pickle.dump(model_n, open(save_model_name, 'wb'))
            plot_parity(y_pred, y_valid, save_model_name)
            plot_learning_curve(model_n, scaled_X_train, y, save_model_name)
            

if __name__ == "__main__" :
    
    methods = ["HF", "EMP"]
    cutoffs = ["0.0", "7.5"]
    valid = 0.2 
    for method in methods:
        for cutoff in cutoffs:
            save_models(df_id=cutoff, valid=valid, method=method)
