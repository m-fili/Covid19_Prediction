import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold





def train_lasso(X, y, scoring, n_cv=5, random_state=110, multi_output=False):
    
    cv = KFold(n_splits=n_cv)
    
    if multi_output:
        params_lasso = {'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
                        'estimator__selection': ['cyclic', 'random']}
        lasso = MultiOutputRegressor(Lasso(max_iter=1e6, random_state=random_state))
    else:
        params_lasso = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
                        'selection': ['cyclic', 'random']}
        lasso = Lasso(max_iter=1e6, random_state=random_state)
    
    grid = GridSearchCV(lasso, param_grid=params_lasso, scoring=scoring, n_jobs=-1, cv=cv)
    grid.fit(X, y)
    
    return grid





def train_ridge(X, y, scoring, n_cv=5, random_state=110, multi_output=False):
    
    cv = KFold(n_splits=n_cv)
    
    if multi_output:
        ridge = MultiOutputRegressor(Ridge(max_iter=1e6, random_state=random_state))
        params_ridge = {'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    else:
        ridge = Ridge(max_iter=1e6, random_state=random_state)
        params_ridge = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    
    grid = GridSearchCV(ridge, param_grid=params_ridge, scoring=scoring, n_jobs=-1, cv=cv)    
    grid.fit(X, y)
    
    return grid