'''
Class for encoding ATP match data into numpy-form training data.

Current data format:
    - Better ranked player name (one-hot)
    - Better player rank (single float)
    - Better player age (single float)
    ...
    [above features repeated for worse ranked player]
    ...
    - Tournament name (one-hot)
    - Binary label, 1 if better ranked player won
'''

import numpy as np
import pandas as pd
import pickle

class PlayerDataEncoder:
    def __init__(self):
        # Dicts that map categorical feature to
        # one-hot numpy array.
        self.players = {}
        self.tournaments = {}

    def create_one_hot(self, list_data):
        one_hot_dict = {}
        unique = list(set(list_data))
        for i, name in enumerate(unique):
            vec = np.zeros((len(unique),))
            vec[i] = 1
            one_hot_dict[name.lower()] = vec
        return one_hot_dict

    def create_example(self, f_data, s_data, t_name):
        full_input = [
                self.players[f_data[0].lower()],
                np.array([f_data[1]]),
                np.array([f_data[2]]),
                self.players[s_data[0].lower()],
                np.array([s_data[1]]),
                np.array([s_data[2]]),
                self.tournaments[t_name.lower()]]
        return full_input

    def generate_train(self, start_year, end_year, train_path):
        if start_year < 2010 or end_year > 2019:
            print("Invalid years.")
            return

        def drop_mid(name):
            split_name = name.split()
            return ' '.join((split_name[0], split_name[-1]))

        frames = []
        for year in range(start_year, end_year + 1):
            df = pd.read_csv('~/Tennis/tennis_atp/atp_matches_' + str(year) + '.csv')
            # Drop middle names of players.
            df['winner_name'] = df['winner_name'].apply(drop_mid)
            df['loser_name'] = df['loser_name'].apply(drop_mid)
            # Drop davis cup.
            frames.append(df[df['tourney_name'].map(lambda x: 'Davis' not in x)])
        full_frame = pd.concat(frames)

        self.players = self.create_one_hot(list(full_frame['winner_name']) + list(full_frame['loser_name']))
        self.tournaments = self.create_one_hot(list(full_frame['tourney_name']))

        # Writes numpy matrix to train_path consisting of
        # features followed by single binary label.
        data_matrix = []
        for i, row in full_frame.iterrows():
            f_data = list(row[['winner_name', 'winner_rank', 'winner_age']])
            s_data = list(row[['loser_name', 'loser_rank', 'loser_age']])
            t_name = str(row['tourney_name'])
            label = [1] # Better ranked player won.
            if f_data[1] > s_data[1]:
                # If winner rank is worse than loser_rank, swap.
                f_data, s_data = s_data, f_data
                label = [0]
            full_input = self.create_example(f_data, s_data, t_name)
            full_input.append(np.array(label))
            data_matrix.append(np.array([np.concatenate(full_input)]))
        np.savetxt(train_path, np.concatenate(data_matrix, axis=0), delimiter=',') 
        
def main():
    encoder = PlayerDataEncoder()
    encoder.generate_train(2019, 2019, 'train_data.csv')
    pickle.dump(encoder, open('data_encoder.p', 'wb'))

if __name__ == '__main__':
    main()
