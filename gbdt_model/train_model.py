'''
Binary for training and saving GBDT model.
'''

import numpy as np
import pickle
import xgboost

def main():
    train_data = np.genfromtxt('train_data.csv', delimiter=',')
    X, y = train_data[:,:-1], train_data[:,-1]

    model = xgboost.XGBRegressor(max_depth=6)
    model.fit(X, y)

    pickle.dump(model, open('gbdt_model.p', 'wb'))

if __name__ == '__main__':
    main()
