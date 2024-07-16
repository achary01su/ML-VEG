import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from filepaths import filepaths

def load_indi_data(filepath_red, filepath_ox, random_state=10):
    
    """
    Loads and combines feature data from reduced and oxidized states.

    Parameters
    ----------
    filepath_red : list of str
        List containing file paths for the reduced state data at two different cutoffs.
    filepath_ox : list of str
        List containing file paths for the oxidized state data at two different cutoffs.
    random_state : int, optional
        Random state for shuffling the data, by default 10.

    Returns
    -------
    tuple of pd.DataFrame
        Shuffled dataframes containing combined features for both cutoffs.

    Raises
    ------
    ValueError
        If the input file paths are not correctly assigned.
    """
    
    if len(filepath_ox) == 2 and len(filepath_red) == 2:
        red_00 = np.genfromtxt(filepath_red[0], dtype='unicode', delimiter=',')
        df_red_00 = pd.DataFrame(red_00[1:, 1:].astype(float), columns=red_00[0, 1:])
        
        red_75 = np.genfromtxt(filepath_red[1], dtype='unicode', delimiter=',')
        df_red_75 = pd.DataFrame(red_75[1:, 1:].astype(float), columns=red_75[0, 1:])
        
        ox_00 = np.genfromtxt(filepath_ox[0], dtype='unicode', delimiter=',')
        df_ox_00 = pd.DataFrame(ox_00[1:, 1:].astype(float), columns=ox_00[0, 1:])

        ox_75 = np.genfromtxt(filepath_ox[1], dtype='unicode', delimiter=',')
        df_ox_75 = pd.DataFrame(ox_75[1:, 1:].astype(float), columns=ox_75[0, 1:])

        # Combine reduced and oxidized dataframes
        comb_00 = pd.concat([df_red_00, df_ox_00], ignore_index=True)
        comb_75 = pd.concat([df_red_75, df_ox_75], ignore_index=True)

        # Drop columns that might be 0
        cols = comb_00.columns
        for col in cols:
            if (comb_00[col] == 0.0).all():
                comb_00 = comb_00.drop(columns=col)
                comb_75 = comb_75.drop(columns=col)
        
        # Return shiffuffled datadrames, can write them to a file as well
        return shuffle(comb_00, random_state=random_state), shuffle(comb_75, random_state=random_state)
    
    else:
        raise ValueError("Enter correct filepaths")


def load_all_data(df_num='0.0', validation=0.1, method='HF', random_state=10):
    
    """
    Loads and combines feature data from all systems, splits into training and testing sets.

    Parameters
    ----------
    df_num : str, optional
        Cutoff value for the data, either '0.0' or '7.5', by default '0.0'.
    validation : float, optional
        Proportion of the data to be used as validation set, by default 0.1.
    method : str, optional
        Method used for generating the feature lists, either 'HF' or 'EMP', by default 'HF'.
    random_state : int, optional
        Random state for shuffling the data, by default 10.

    Returns
    -------
    tuple of pd.DataFrame
        Training and testing sets for features and target values.

    Raises
    ------
    ValueError
        If the df_num is not '0.0' or '7.5'.
    """
    
    # Load dataframes for all systems
    phen_00, phen_75 = load_indi_data(filepaths('phenol', method=method)[0], filepaths('phenol', method=method)[1])
    phei_00, phei_75 = load_indi_data(filepaths('phenolate', method=method)[0], filepaths('phenolate', method=method)[1])
    benz_00, benz_75 = load_indi_data(filepaths('benzene', method=method)[0], filepaths('benzene', method=method)[1])
    indo_00, indo_75 = load_indi_data(filepaths('indole', method=method)[0], filepaths('indole', method=method)[1])
    lumi_00, lumi_75 = load_indi_data(filepaths('lumiflavin', method=method)[0], filepaths('lumiflavin', method=method)[1])

    # Determine length of validation set and extract it separately 
    valid_phen = int(len(phen_00) * validation)
    valid_phei = int(len(phei_00) * validation)
    valid_benz = int(len(benz_00) * validation)
    valid_indo = int(len(indo_00) * validation)
    valid_lumi = int(len(lumi_00) * validation)

    df_00_list = [phen_00[:-valid_phen], phei_00[:-valid_phei], benz_00[:-valid_benz],
                  indo_00[:-valid_indo], lumi_00[:-valid_lumi]]

    df_00_list_test = [phen_00[-valid_phen:], phei_00[-valid_phei:], benz_00[-valid_benz:],
                       indo_00[-valid_indo:], lumi_00[-valid_lumi:]]

    df_75_list = [phen_75[:-valid_phen], phei_75[:-valid_phei], benz_75[:-valid_benz],
                  indo_75[:-valid_indo], lumi_75[:-valid_lumi]]

    df_75_list_test = [phen_75[-valid_phen:], phei_75[-valid_phei:], benz_75[-valid_benz:],
                       indo_75[-valid_indo:], lumi_75[-valid_lumi:]]

    # Combine train and validation separately and shuffle again
    comb_df_75 = pd.concat(df_75_list, ignore_index=True)
    comb_df_75_test = pd.concat(df_75_list_test, ignore_index=True)

    comb_df_00 = pd.concat(df_00_list, ignore_index=True)
    comb_df_00_test = pd.concat(df_00_list_test, ignore_index=True)

    shuffle_comb_75 = shuffle(comb_df_75, random_state=random_state)
    shuffle_comb_75_test = shuffle(comb_df_75_test, random_state=random_state)
    
    shuffle_comb_00 = shuffle(comb_df_00, random_state=random_state)
    shuffle_comb_00_test = shuffle(comb_df_00_test, random_state=random_state)

    # Convert delta E to eV from au 
    y_train = shuffle_comb_75.iloc[:,-1] * 27.2114
    y_test = shuffle_comb_75_test.iloc[:,-1] * 27.2114
    
    # Assign feature list from cutoff
    if df_num == '0.0':
        X_train = shuffle_comb_00.iloc[:,:-1]
        X_test = shuffle_comb_00_test.iloc[:,:-1]
    elif df_num == '7.5':
        X_train = shuffle_comb_75.iloc[:,:-1]
        X_test = shuffle_comb_75_test.iloc[:,:-1]
    else:
        raise ValueError("Enter either '0.0' or '7.5' for df_num")
    
    return (X_train, X_test, y_train, y_test)


def load_system_data(system, df_num='0.0', validation=0.1, method='HF'):
    
    """
    Loads feature data for a specific system, splits into training and testing sets.

    Parameters
    ----------
    system : str
        The chemical system for which data is required.
        Valid options are 'phenol', 'phenolate', 'benzene', 'indole', 'lumiflavin'.
    df_num : str, optional
        Cutoff value for the data, either '0.0' or '7.5', by default '0.0'.
    validation : float, optional
        Proportion of the data to be used as validation set, by default 0.1.
    method : str, optional
        Method used for generating the feature lists, either 'HF' or 'EMP', by default 'HF'.

    Returns
    -------
    tuple of pd.DataFrame
        Training and testing sets for features and target values.
    
    Raises
    ------
    ValueError
        If the system name is not in the list of valid systems.
        If the df_num is not '0.0' or '7.5'.
    """
    
    sys_list = ['phenol', 'phenolate', 'benzene', 'indole', 'lumiflavin']
    
    if system not in sys_list:
        raise ValueError(f"Invalid system name. Valid options are: {', '.join(sys_list)}")
    
    sys_00, sys_75 = load_indi_data(filepaths(system, method=method)[0], filepaths(system, method=method)[1])
        
    y = sys_75.iloc[:,-1] * 27.2114
    if df_num == '0.0':
        X = sys_00.iloc[:,:-1]
    elif df_num == '7.5':
        X = sys_75.iloc[:,:-1]
    else:
        raise ValueError("Enter either '0.0' or '7.5' to get corresponding df")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation, random_state=20)
    
    return (X_train, X_test, y_train, y_test)
