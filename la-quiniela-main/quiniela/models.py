import pickle
import time
import sqlite3
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report  # conf matrix
from sklearn.feature_selection import RFE
import seaborn as sns  # conf matrix
from sklearn.feature_selection import chi2
from time import time
import datetime
import warnings

warnings.filterwarnings("ignore")
from quiniela import io
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler


def fix_date(date):
    if date.year > 2021:
        year = date.year - 100
    else:
        year = date.year
    return datetime.date(int(year), date.month, date.day)


def parse_score(score):
    w = ''
    if score:
        score = score.split(':')
        if score[0] > score[1]:
            w = 1
        elif score[0] < score[1]:
            w = 2
        elif score[0] == score[1]:
            w = 0
    else:
        w = 'Unknown'
    return w


def last_aux(row, num, dict_teams, pred=False):
    h_t = row['home_id']
    a_t = row['away_id']
    date = row['date']

    # get all matches of each team
    matches_h = dict_teams[h_t]
    matches_a = dict_teams[a_t]

    # get last matches of each team
    last_matches_h = matches_h[matches_h['date'] < date].sort_values(by='date', ascending=False).iloc[0:num, :]
    last_matches_a = matches_a[matches_a['date'] < date].sort_values(by='date', ascending=False).iloc[0:num, :]

    if ((len(last_matches_h) == num) and (len(last_matches_a) == num)):
        # column of away goals, home goals, home victories, away victories
        team_home_hg = last_matches_h['home_goals'][last_matches_h['home_id'] == h_t].sum()
        team_home_ag = last_matches_h['away_goals'][last_matches_h['away_id'] == h_t].sum()
        team_away_hg = last_matches_a['home_goals'][last_matches_a['home_id'] == a_t].sum()
        team_away_ag = last_matches_a['away_goals'][last_matches_a['away_id'] == a_t].sum()

        team_home_vh = (last_matches_h['Result'] == 1)[last_matches_h['home_id'] == h_t].sum(axis=0)
        team_home_va = (last_matches_h['Result'] == 2)[last_matches_h['away_id'] == h_t].sum(axis=0)
        team_away_vh = (last_matches_a['Result'] == 2)[last_matches_a['away_id'] == a_t].sum(axis=0)
        team_away_va = (last_matches_a['Result'] == 1)[last_matches_a['home_id'] == a_t].sum(axis=0)

        row['FHG'] = int(team_home_hg + team_home_ag)
        row['FAG'] = int(team_away_hg + team_away_ag)
        row['VHT'] = int(team_home_vh + team_home_va)
        row['VAT'] = int(team_away_vh + team_away_va)
    return row


# This function gets some features for the last num matches for each team
def last_matches_opt(df, num, pred=False):
    dict_teams = {}
    for team in df['home_id'].unique():
        dict_teams[team] = df[(df['home_id'] == team) | (df['away_id'] == team)]

    return df.apply(lambda row: last_aux(row, num, dict_teams), axis=1)


def last_dir_aux(row, num, dict_teams, pred=False):
    h_t = row['home_id']
    a_t = row['away_id']
    date = row['date']

    matches_h = dict_teams[h_t]
    matches_a = dict_teams[a_t]
    matches_all = pd.concat([matches_h, matches_a])

    # get last matches of direct matches
    last_matches = matches_all[matches_all['date'] < date].sort_values(by='date', ascending=False).iloc[0:num, :]
    if (len(last_matches) == num):
        # column of away goals, home goals, home victories, away victories
        total_goals = last_matches['total_goals'].sum()
        diff_goals = last_matches['goal_difference'].sum()

        team_home_hg = last_matches['home_goals'][last_matches['home_id'] == h_t].sum()
        team_home_ag = last_matches['away_goals'][last_matches['away_id'] == h_t].sum()
        team_away_hg = last_matches['home_goals'][last_matches['home_id'] == a_t].sum()
        team_away_ag = last_matches['away_goals'][last_matches['away_id'] == a_t].sum()

        team_home_vh = (last_matches['Result'] == 1)[last_matches['home_id'] == h_t].sum(axis=0)
        team_home_va = (last_matches['Result'] == 2)[last_matches['away_id'] == h_t].sum(axis=0)
        team_away_vh = (last_matches['Result'] == 2)[last_matches['away_id'] == a_t].sum(axis=0)
        team_away_va = (last_matches['Result'] == 1)[last_matches['home_id'] == a_t].sum(axis=0)

        row['FTG_dm'] = int(total_goals)
        row['FDG_dm'] = int(diff_goals)
        row['FHG_dm'] = int(team_home_hg + team_home_ag)
        row['FAG_dm'] = int(team_away_hg + team_away_ag)
        row['VHT_dm'] = int(team_home_vh + team_home_va)
        row['VAT_dm'] = int(team_away_vh + team_away_va)

    return row


# This function gets some features for the last num matches for each team
def last_dir_matches_opt(df, num, pred=False):
    dict_teams = {}
    for team in df['home_id'].unique():
        dict_teams[team] = df[(df['home_id'] == team) | (df['away_id'] == team)]

    return df.apply(lambda row: last_dir_aux(row, num, dict_teams), axis=1)


def last_matches_opt_pred(df, num, df_full):
    dict_teams = {}
    for team in df_full['home_id'].unique():
        dict_teams[team] = df_full[(df_full['home_id'] == team) | (df_full['away_id'] == team)]

    return df.apply(lambda row: last_aux(row, num, dict_teams), axis=1)


def last_dir_matches_opt_pred(df, num, df_full):
    dict_teams = {}
    for team in df_full['home_id'].unique():
        dict_teams[team] = df_full[(df_full['home_id'] == team) | (df_full['away_id'] == team)]

    return df.apply(lambda row: last_dir_aux(row, num, dict_teams), axis=1)


def Pearson_select(X, y, num_features):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation for each feature
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # feature name
    feature_selection = X.iloc[:, np.argsort(np.abs(cor_list))[-num_features:]].columns.tolist()
    # feature selection
    cor_support = [True if i in feature_selection else False for i in feature_name]
    return cor_list, feature_selection


def process_data(df_aux, self):
    # change date format
    df_aux['date'] = pd.to_datetime(df_aux['date']).dt.date
    # fix date
    df_aux['date'] = df_aux['date'].apply(fix_date)

    # Match results
    scores = df_aux['score'].values
    scores = [x.split(':') if x else [-1, -1] for x in scores]
    df_aux['Result'] = df_aux['score'].apply(parse_score)

    # Home goals and away goals
    home_goals = []
    away_goals = []
    for scored_goals in scores:
        home_goals.append(scored_goals[0])
        away_goals.append(scored_goals[1])
    df_aux['home_goals'] = pd.to_numeric(home_goals)
    df_aux['away_goals'] = pd.to_numeric(away_goals)

    # Goal difference
    df_aux['goal_difference'] = abs(df_aux['home_goals'] - df_aux['away_goals'])
    df_aux['total_goals'] = df_aux['home_goals'] + df_aux['away_goals']

    # Assigning values to each team in order of aparison (encode the label)
    teams = df_aux['home_team'].drop_duplicates().values
    df_aux['home_id'] = 0
    df_aux['away_id'] = 0
    for i in range(len(teams)):
        df_aux.loc[df_aux['home_team'] == teams[i], ['home_id']] = i
        df_aux.loc[df_aux['away_team'] == teams[i], ['away_id']] = i
        self.dic_teams[teams[i]] = i

    # droping unknown
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['Result'])
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['date'])
    df_aux.reset_index(drop=True, inplace=True)  # reset index after droping the unknown

    # new columns for features
    # last direct matches
    df_aux['FHG_dm'] = np.nan  # Feature home goals (last direct matches)
    df_aux['FAG_dm'] = np.nan  # Feature away goals (last direct matches)
    df_aux['FTG_dm'] = np.nan  # Feature total goals (last direct matches)
    df_aux['FDG_dm'] = np.nan  # Feature difference goals (last direct matches)
    df_aux['VHT_dm'] = np.nan  # victories home team ( last direct matches)
    df_aux['VAT_dm'] = np.nan  # victories away team ( last direct matches)

    # last matches of each team
    df_aux['FHG'] = np.nan  # Feature home goals (last matches)
    df_aux['FAG'] = np.nan  # Feature away goals (last matches)
    df_aux['VHT'] = np.nan  # victories home team ( last matches)
    df_aux['VAT'] = np.nan  # victories away team ( last matches)

    df_aux = last_matches_opt(df_aux, 5, True)
    df_aux = last_dir_matches_opt(df_aux, 7, True)

    # dropping unknown
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['FHG_dm'])
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['FHG'])

    # reset index after droping the unknown
    df_aux.reset_index(drop=True, inplace=True)

    df_aux['FHG_dm'] = df_aux['FHG_dm'].astype('int')
    df_aux['FAG_dm'] = df_aux['FAG_dm'].astype('int')
    df_aux['FTG_dm'] = df_aux['FTG_dm'].astype('int')
    df_aux['FDG_dm'] = df_aux['FDG_dm'].astype('int')
    df_aux['VHT_dm'] = df_aux['VHT_dm'].astype('int')
    df_aux['VAT_dm'] = df_aux['VAT_dm'].astype('int')
    df_aux['FHG'] = df_aux['FHG'].astype('int')
    df_aux['FAG'] = df_aux['FAG'].astype('int')
    df_aux['VAT'] = df_aux['VAT'].astype('int')
    df_aux['VHT'] = df_aux['VHT'].astype('int')

    return df_aux, self.dic_teams


def process_data_predict(df_aux, df_full, dic_teams):
    # change date format
    df_aux['date'] = pd.to_datetime(df_aux['date']).dt.date
    # fix date
    df_aux['date'] = df_aux['date'].apply(fix_date)

    # Assigning values to each team in order of aparison (encode the label)
    teams = df_aux['home_team'].drop_duplicates().values
    df_aux['home_id'] = 0
    df_aux['away_id'] = 0
    for i in range(len(teams)):
        df_aux.loc[df_aux['home_team'] == teams[i], ['home_id']] = dic_teams[teams[i]]
        df_aux.loc[df_aux['away_team'] == teams[i], ['away_id']] = dic_teams[teams[i]]

    # droping unknown
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['date'])
    df_aux.reset_index(drop=True, inplace=True)  # reset index after droping the unknown

    # new columns for features
    # last direct matches
    df_aux['FHG_dm'] = np.nan  # Feature home goals (last direct matches)
    df_aux['FAG_dm'] = np.nan  # Feature away goals (last direct matches)
    df_aux['FTG_dm'] = np.nan  # Feature total goals (last direct matches)
    df_aux['FDG_dm'] = np.nan  # Feature difference goals (last direct matches)
    df_aux['VHT_dm'] = np.nan  # victories home team ( last direct matches)
    df_aux['VAT_dm'] = np.nan  # victories away team ( last direct matches)

    # last matches of each team
    df_aux['FHG'] = np.nan  # Feature home goals (last matches)
    df_aux['FAG'] = np.nan  # Feature away goals (last matches)
    df_aux['VHT'] = np.nan  # victories home team ( last matches)
    df_aux['VAT'] = np.nan  # victories away team ( last matches)

    df_aux = last_matches_opt_pred(df_aux, 5, df_full)
    df_aux = last_dir_matches_opt_pred(df_aux, 7, df_full)

    # dropping unknown
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['FHG_dm'])
    df_aux = df_aux.mask(df_aux.eq('Unknown')).dropna(subset=['FHG'])

    # reset index after droping the unknown
    df_aux.reset_index(drop=True, inplace=True)

    df_aux['FHG_dm'] = df_aux['FHG_dm'].astype('int')
    df_aux['FAG_dm'] = df_aux['FAG_dm'].astype('int')
    df_aux['FTG_dm'] = df_aux['FTG_dm'].astype('int')
    df_aux['FDG_dm'] = df_aux['FDG_dm'].astype('int')
    df_aux['VHT_dm'] = df_aux['VHT_dm'].astype('int')
    df_aux['VAT_dm'] = df_aux['VAT_dm'].astype('int')
    df_aux['FHG'] = df_aux['FHG'].astype('int')
    df_aux['FAG'] = df_aux['FAG'].astype('int')
    df_aux['VAT'] = df_aux['VAT'].astype('int')
    df_aux['VHT'] = df_aux['VHT'].astype('int')

    return df_aux


class QuinielaModel:
    model = None
    processed_df = None
    dic_teams = {}

    def train(self, train_data):
        df_aux = train_data
        df_aux, self.dic_teams = process_data(df_aux, self)
        self.processed_df = df_aux.copy()
        # Features precomputated
        features = ['FHG_dm', 'FAG_dm', 'FHG', 'FAG']
        target = 'Result'
        X = df_aux[features]
        y = df_aux[target]

        self.model = LogisticRegression(multi_class='ovr', max_iter=300, class_weight="balanced")
        self.model.fit(X, y)

        pass

    def predict(self, predict_data):
        # Do something here to predict
        # self.model = QuinielaModel.load('../models/quiniela.model')
        predict_data = process_data_predict(predict_data, self.processed_df, self.dic_teams)
        features = ['FHG_dm', 'FAG_dm', 'FHG', 'FAG']
        y = predict_data[features]
        y_pred = self.model.predict(y)
        y_pred_formatted = []
        for pred in y_pred:
            if pred == 0:
                y_pred_formatted.append('X')
            else:
                y_pred_formatted.append(str(pred))
        return y_pred_formatted

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)



