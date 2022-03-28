import numpy as np
import pandas as pd




def filter_country(df, country, COLS, start_time=None, end_time=None, outcome='NewCases'):
        
    df_country = df.loc[df.CountryName == country].sort_index(ascending=True)
    df_country.set_index('Date', inplace=True)
    
    if (start_time is not None) & (end_time is not None):
        df_country = df_country.loc[start_time:end_time]
    
    df_country.index.seq = 'D'
    
    X, y = df_country[COLS + [outcome]].values, df_country[outcome].values.reshape(-1, 1)
    indx = df_country.index
    
    return X, y, indx




def colnames_generator(columns, T=10):
    """
    Creates the column names for the new time-serie dataset.
    
    Parameters
    ----------
    columns: list
        list of initial column names
    T: int
        a value indicating the number past days to include
        
    Returns
    -------
    A list of new columns names.
    """
    
    new_cols = []
    
    for i in range(T):
        for j in columns:
            new_cols.append(j + f'(t-{T-i-1})')
    
    return np.array(new_cols)





def Create_TrainTest(X, y, method='single', indx=None, T=10, n_test=7):
    """
    Creates Train test split for different scenarios.
    
    Parameters
    ----------
    X: np.array
        Array of original features
    y: np.array
        Array of original outcomes
    methods: 'single' or 'multi'
        single: Suitable for 1-step ahead or h-step ahead (incrementally: subsituting t+1 with the prediction after each
        prediction).
        multi: Suitable for h-step ahead all at once (multi-output).
    indx: np.array, pd.Series
        indices for each time point. If "None", it will create a range of length as input X (1, 2, .., N).
    T: int > 0
        Number of days from past to include in the feature set
    n_test: int > 0
        if (method = single): number of days to consider in test set.
        if (method = multi): equivalent to h in h-step ahead prediction. (e.g., if 3 then it creates t+1, t+2, t+3 in 
        test set).
    
    
    Return
    ------
    Xtrain, Xtest, Ytrain, Ytest, TrainIndx, TestIndx
        (Xtrain, Ytrain): The new training set.
        (Xtest, Ytest): The new test set.
        (TrainIndx, TestIndx): prediction index (time of prediction for t+1).
    """
    
    assert method in ['single', 'multi'], 'Method should be either "single" or "multi".'
    assert len(X) == len(y), "X and y should have the same length"
    assert T > 0, 'T should be positive'
    assert n_test > 0, 'n_test should be positive'
    
    N = len(X)
    assert N >= T + n_test - 1, 'Large Values! Redefine "T" or "n_test".'
    
    indx_new = []
    
    if indx is None:
        indx = np.arange(N)
    
    if method == 'single':        
        X_new = []
        y_new = []

        for t in range(N-T):
            X_new.append(X[t:t+T,:].flatten())
            y_new.append(y[t+T])
            indx_new.append(indx[t+T])
        
        # Stacking
        X_new = np.stack(X_new, axis=0)
        y_new = np.stack(y_new, axis=0)
        
        # Split
        Xtrain, Ytrain = X_new[:-n_test], y_new[:-n_test]
        Xtest, Ytest = X_new[-n_test:], y_new[-n_test:]
        TrainIndx, TestIndx = indx_new[:-n_test], indx_new[-n_test:]
        
    
    elif method == 'multi':       
        X_new = []
        Y_new = []

        for t in range(N - T - n_test + 1):
            X_new.append(X[t:t+T, :].flatten())
            Y_new.append(y[t+T : t+T+n_test].flatten())
            indx_new.append(indx[t+T : t+T+n_test])

        X_new = np.stack(X_new, axis=0)
        Y_new = np.stack(Y_new, axis=0)
        indx_new = np.stack(indx_new, axis=0)

        # Train & Test Split
        Xtrain, Ytrain = X_new[:-1], Y_new[:-1]
        Xtest, Ytest = X_new[-1:], Y_new[-1:]
        TrainIndx, TestIndx = indx_new[:-1], indx_new[-1]
    
    
    return Xtrain, Xtest, Ytrain, Ytest, TrainIndx, TestIndx






