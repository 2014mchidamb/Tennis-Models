'''
Binary for predicting result of match using trained model.
'''

from data_utils import PlayerDataEncoder
import numpy as np
import pickle

def get_player_data(player_type='Better'):
    p_data = []
    p_data.append(input(player_type + ' name: '))
    p_data.append(input(player_type + ' rank: '))
    p_data.append(input(player_type + ' age: '))
    return p_data

def main():
    data_encoder = pickle.load(open('data_encoder.p', 'rb'))
    print('Possible tournament options: ')
    print(data_encoder.tournaments.keys())

    f_data = get_player_data()
    s_data = get_player_data('Worse')
    t_name = input('Tournament name: ')

    print(data_encoder.create_example(f_data, s_data, t_name))

if __name__ == '__main__':
    main()
