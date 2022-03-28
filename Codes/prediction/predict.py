import numpy as np




def onestep_prediction(Xtest, model):
    """
    Creates 1-step ahead prediction.
    """
    pred = model.predict(X_test)
    
    return pred



def multistep_prediction(Xtest, model, n_test=7):
    """
    It does the rolling one-step ahead prediction. using t-1, t-2, ... t-T to predict t. Then it add the prediction as the 
    value of observation in time point t. Now t predicts t+1 using t, t-1, t-2, ..., t-T+1, and so forth.
    
    Parameter
    ---------
    X: np.array
        Features
    model: sklearn model
        A model that you want to use for prediction
    n_test: int > 0
        Number of periods in future that you want to predict (Equivalent to h-step). This is typically as the same size 
        of the test set.
    
    Return
    ------
    np.array: An array of prediction results
    """
    
    # multi-step forecast
    multistep_predictions = []

    # first test input
    last_x = Xtest[0]

    while len(multistep_predictions) < n_test:
        pred = model.predict(last_x.reshape(1, -1))[0]

        # update the predictions list
        multistep_predictions.append(pred)

        # make the new input
        last_x = np.roll(last_x, -1)
        last_x[-1] = pred
    
    return np.array(multistep_predictions).reshape(-1, 1)

