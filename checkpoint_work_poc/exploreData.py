import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.datasets import metabric
import pycox as pycox
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pycox.evaluation import EvalSurv
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Load dataset


# load and transform data
# dfmatch = pd.read_csv('~/Downloads/opendota500k/match.csv')
# dfplayers = pd.read_csv('~/Downloads/opendota500k/players.csv')
# dfplayerratings = pd.read_csv('~/Downloads/opendota500k/player_ratings.csv')
# dfplayertime = pd.read_csv('~/Downloads/opendota500k/player_time.csv')
dfobjectives = pd.read_csv('~/Downloads/opendota500k/objectives.csv')

print(dfobjectives.head(5))

# exit()


# print('loaded datasets')

# dfplayerratings['win_rate'] = dfplayerratings['total_wins'] / dfplayerratings['total_matches']

# percentile_exp_90 = dfplayerratings['total_matches'].quantile(0.9)

# dfplayerratings['high_exp'] = dfplayerratings['total_matches'].apply(lambda x: 10 if x > percentile_exp_90 else 0)

# dfplayersmatch = pd.merge(dfplayers[['match_id', 'account_id', 'player_slot']], dfplayerratings[['account_id', 'win_rate', 'high_exp', 'trueskill_mu', 'trueskill_sigma']], on='account_id', how='left')


# def getTeam(row):
#     if row['player_slot'] >= 100:
#         return 'team_100'
#     else:
#         return 'team_1'

# dfplayersmatch['team'] = dfplayersmatch.apply(getTeam, axis=1)

# #flatten by match and make _t100 columns
# # Define columns to average
# cols_to_average = ['win_rate', 'high_exp', 'trueskill_mu']

# # Group by 'Category' and 'Subcategory', then average specified columns
# dfteamsmatch = dfplayersmatch.groupby(['match_id', 'team'])[cols_to_average].mean().reset_index()



# dfteam1 = dfteamsmatch[dfteamsmatch['team'] == 'team_1']
# dfteam100 = dfteamsmatch[dfteamsmatch['team'] == 'team_100']

# dfteam100 = dfteam100.add_suffix('_t100')
# dfteam100.rename(columns={'match_id_t100':'match_id'}, inplace=True)

# print(dfteam1.head(3))

# print(dfteam100.head(3))

# dfplayersmatchmerged = dfteam1.merge(dfteam100, on="match_id")


# df = pd.merge(dfmatch[['match_id', 'first_blood_time', 'radiant_win', 'duration']], dfplayersmatchmerged[['match_id', 'win_rate', 'high_exp', 'trueskill_mu', 'win_rate_t100', 'high_exp_t100', 'trueskill_mu_t100']], on='match_id', how='left')


# # Standardize data
# scaler = StandardScaler()

# df[['first_blood_time', 'win_rate', 'high_exp', 'trueskill_mu', 'win_rate_t100', 'high_exp_t100', 'trueskill_mu_t100']] = scaler.fit_transform(df[['first_blood_time', 'win_rate', 'high_exp', 'trueskill_mu', 'win_rate_t100', 'high_exp_t100', 'trueskill_mu_t100']])


# print('created dataset')

# print(df.head(10))

# df.to_csv('~/Downloads/opendota500k/dataset.csv', index=True)

# # else :
# #     df = pd.read_csv('~/Downloads/opendota500k/dataset.csv')


# # test/train split
# train, test = train_test_split(df, test_size=0.2, random_state=42)
# val, test = train_test_split(test, test_size=0.5, random_state=42)

# # Define features and labels
# cols_standardize = ['first_blood_time', 'win_rate', 'high_exp', 'trueskill_mu', 'win_rate_t100', 'high_exp_t100', 'trueskill_mu_t100']
# cols_categorical = []
# num_features = len(cols_standardize) + len(cols_categorical)
# duration_col = 'duration'
# event_col = 'radiant_win'


# standardize = [([col], StandardScaler()) for col in cols_standardize]
# leave = [(col, None) for col in cols_categorical]



# # Data transformations
# x_train = train[cols_standardize + cols_categorical].values.astype('float32')
# y_train = train[duration_col].values.astype('float32')
# event_train = train[event_col].values

# x_val = val[cols_standardize + cols_categorical].values.astype('float32')
# y_val = val[duration_col].values.astype('float32')
# event_val = val[event_col].values

# x_test = test[cols_standardize + cols_categorical].values.astype('float32')
# y_test = test[duration_col].values.astype('float32')
# event_test = test[event_col].values
