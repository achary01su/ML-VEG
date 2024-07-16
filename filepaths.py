def filepaths(system, method='HF'):
    """
    Returns the file paths for the given system and method.

    Parameters
    ----------
    system : str
        The chemical system for which file paths are required.
        Valid options are 'phenol', 'phenolate', 'benzene', 'indole', 'lumiflavin'.
    method : str, optional
        The method used for generating the feature lists. 
        Valid options are 'HF' (Hartree-Fock) and 'EMP' (semi-empirical method). 
        Default is 'HF'.

    Returns
    -------
    list of list of str
        A list of lists containing file paths for the feature lists of the given system 
        and method.

    Raises
    ------
    ValueError
        If the method provided is not 'HF' or 'EMP'.
    """
    hf_paths = {'phenol' : [["phenol/feature_list_0.0.dat","phenol/feature_list_7.5.dat"],
                        ["phenol_ionized/feature_list_0.0.dat", "phenol_ionized/feature_list_7.5.dat"]],
            'phenolate' : [["phenolate/feature_list_0.0.dat","phenolate/feature_list_7.5.dat"],
                        ["phenolate_radical/feature_list_0.0.dat", "phenolate_radical/feature_list_7.5.dat"]],
            'benzene' : [["benzene/feature_list_0.0.dat","benzene/feature_list_7.5.dat"],
                        ["benzene_ionized/feature_list_0.0.dat", "benzene_ionized/feature_list_7.5.dat"]],
            'indole':   [["indole/feature_list_0.0.dat","indole/feature_list_7.5.dat"],
                        ["indole_ionized/feature_list_0.0.dat", "indole_ionized/feature_list_7.5.dat"]],
            'lumiflavin' :  [["lumiflavin/feature_list_0.0.dat","lumiflavin/feature_list_7.5.dat"], 
                         ["lumiflavin_anion/feature_list_0.0.dat","lumiflavin_anion/feature_list_7.5.dat" ]]
            }
    
    emp_paths = {'phenol' : [["phenol/EMP1_feature_list_0.0.dat","phenol/EMP1_feature_list_7.5.dat"],
                        ["phenol_ionized/EMP1_feature_list_0.0.dat", "phenol_ionized/EMP1_feature_list_7.5.dat"]],
            'phenolate' : [["phenolate/EMP1_feature_list_0.0.dat","phenolate/EMP1_feature_list_7.5.dat"],
                        ["phenolate_radical/EMP1_feature_list_0.0.dat", "phenolate_radical/EMP1_feature_list_7.5.dat"]],
            'benzene' : [["benzene/EMP1_feature_list_0.0.dat","benzene/EMP1_feature_list_7.5.dat"],
                        ["benzene_ionized/EMP1_feature_list_0.0.dat", "benzene_ionized/EMP1_feature_list_7.5.dat"]],
            'indole':   [["indole/EMP1_feature_list_0.0.dat","indole/EMP1_feature_list_7.5.dat"],
                        ["indole_ionized/EMP1_feature_list_0.0.dat", "indole_ionized/EMP1_feature_list_7.5.dat"]],
            'lumiflavin': [["lumiflavin/EMP1_feature_list_0.0.dat","lumiflavin/EMP1_feature_list_7.5.dat"], 
                         ["lumiflavin_anion/EMP1_feature_list_0.0.dat","lumiflavin_anion/EMP1_feature_list_7.5.dat"]]
            }
    
    if method == 'HF':
        return hf_paths[system]
    elif method == 'EMP':
        return emp_paths[system]
    else:
        raise ValueError("Incorrect method: Enter either 'HF' or 'EMP' ")
    