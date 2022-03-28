import matplotlib.pyplot as plt




def plot(train_indx, y_train, test_indx, y_test, pred_train, pred_test, name):
    
    plt.figure(figsize=(12, 4))

    plt.plot(train_indx, y_train, color='b')
    plt.plot(test_indx, y_test, color='g')

    plt.plot(train_indx, pred_train, color='r', linestyle='dashed')
    plt.plot(test_indx, pred_test, color='purple', linestyle='dashed')
    
    plt.title(name, size=14, fontfamily='serif')