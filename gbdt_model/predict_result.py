'''
Binary for predicting result of match using trained model.
'''

from data_utils import PlayerDataEncoder
from xgboost import XGBRegressor
import numpy as np
import pickle

def get_player_data(player_type='Better'):
    p_data = []
    p_data.append(input(player_type + ' ranked player name: '))
    p_data.append(input(player_type + ' ranked player rank: '))
    p_data.append(input(player_type + ' ranked player age: '))
    return p_data

def main():
    data_encoder = pickle.load(open('data_encoder.p', 'rb'))
    model = pickle.load(open('gbdt_model.p', 'rb'))

    print('Possible tournament options: ')
    print(data_encoder.tournaments.keys())

    f_data = get_player_data()
    s_data = get_player_data('Worse')
    t_name = input('Tournament name: ')

    example = data_encoder.create_example(f_data, s_data, t_name)
    res = model.predict(np.array([np.concatenate(example)]))
    print('Likelihood that', f_data[0], 'wins: ', res)

if __name__ == '__main__':
    main()
