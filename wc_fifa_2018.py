
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from blist import sortedset


# In[2]:


def create_feature(columns, world_cup_rankings, home, away):
    row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, False]]), columns=columns)
    home_rank = world_cup_rankings.loc[home, 'rank']
    home_points = world_cup_rankings.loc[home, 'weighted_points']
    opp_rank = world_cup_rankings.loc[away, 'rank']
    opp_points = world_cup_rankings.loc[away, 'weighted_points']
    row['average_rank'] = (home_rank + opp_rank) / 2
    row['rank_difference'] = home_rank - opp_rank
    row['point_difference'] = home_points - opp_points
    row['is_friendly'] = False
    return row
    


# Input data files are available in the data folder.

# In[3]:


print(os.listdir('./data'))


# The datasets used here are
# 
# 1. FIFA rankings for all the teams for the last 25 years
# 
# 2. FIFA world cup 2018 schedule
# 
# 3. All international soccer matches played for the last 170 years

# In[4]:


# read the data files and do necessary corrections
rankings = pd.read_csv('./data/fifa_rankings.csv')
rankings = rankings.loc[:, ['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted',
                            'two_year_ago_weighted', 'three_year_ago_weighted', 'rank_date']]
rankings = rankings.replace({'IR Iran': 'Iran'})
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
rankings['weighted_points'] = (rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted']
                                + rankings['three_year_ago_weighted'])/3

world_cup = pd.read_csv('./data/fifa_wc_2018.csv')
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup.columns = ['team', 'group', 'first_match_against', 'second_match_against', 'third_match_against']
world_cup = world_cup.dropna(how='all') # One NA entry is there, don't know why?
world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
world_cup = world_cup.set_index('team')

matches = pd.read_csv('./data/international_soccer_matches.csv')
matches = matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])


# Let's join matches with ranks for different teams

# In[5]:


# rankings for everyday
rankings = rankings.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D')    .first().fillna(method='ffill').reset_index()

# join the ranks
matches = matches.merge(rankings, left_on=['date', 'home_team'], right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings, left_on=['date', 'away_team'],
                        right_on=['rank_date', 'country_full'], suffixes=('_home', '_away'))


# The features we are considering here are
# 
# 1. The rank difference between teams
# 
# 2. Average point difference between teams
# 
# 3. Average rank of the teams
# 
# 4. Whether the match is friendly

# In[6]:


# feature generation
matches['rank_difference'] = (matches['rank_home'] - matches['rank_away'])
matches['average_rank'] = (matches['rank_home'] + matches['rank_away'])/2
matches['point_difference'] = (matches['weighted_points_home'] - matches['weighted_points_away'])
matches['score_difference'] = matches['home_score'] - matches['away_score']
matches['is_won'] = matches['score_difference'] > 0 # take draw as lost
matches['is_friendly'] = matches['tournament'] == 'Friendly'


# We are fitting logistic regression here, which is giving around 69% accuracy.

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = matches.loc[:,['average_rank', 'rank_difference', 'point_difference', 'is_friendly']], matches['is_won']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_clf = LogisticRegression()
model = random_forest_clf.fit(X_train, y_train)
y_pred = random_forest_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[8]:


# let's define a small margin when we safer to predict draw then win
margin = 0.05

# let's define the rankings at the time of the World Cup
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                  rankings['country_full'].isin(world_cup.index.unique())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])


# Let's find out the prediction, irrespective of current results we know now.

# In[9]:


from itertools import combinations

opponents = ['first_match_against', 'second_match_against', 'third_match_against']

world_cup['points'] = 0
world_cup['total_prob'] = 0

for group in sortedset(world_cup['group']):
    print('Starting group {}: '.format(group))
    for home, away in combinations(world_cup.query('group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        row = create_feature(X_test.columns, world_cup_rankings, home, away)
        
        home_win_prob = model.predict_proba(row)[:,1][0]
        world_cup.loc[home, 'total_prob'] += home_win_prob
        world_cup.loc[away, 'total_prob'] += 1-home_win_prob
        
        points = 0
        if home_win_prob <= 0.5 - margin:
            print("{} wins with {:.2f}".format(away, 1-home_win_prob))
            world_cup.loc[away, 'points'] += 3
        if home_win_prob > 0.5 - margin:
            points = 1
        if home_win_prob >= 0.5 + margin:
            points = 3
            world_cup.loc[home, 'points'] += 3
            print("{} wins with {:.2f}".format(home, home_win_prob))
        if points == 1:
            print("Draw")
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1


# In[10]:


pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]

world_cup = world_cup.sort_values(by=['group', 'points', 'total_prob'], ascending=False).reset_index()
next_round_wc = world_cup.groupby('group').nth([0, 1]) # select the top 2
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('team')

finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

labels = list()
odds = list()

for f in finals:
    print("Starting of the {}".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home, away), end='')
        print("{} vs. {}: ".format(home, away), end='')
        row = create_feature(X_test.columns, world_cup_rankings, home, away)

        home_win_prob = model.predict_proba(row)[:,1][0]
        if model.predict_proba(row)[:,1] <= 0.5:
            print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            winners.append(away)
        else:
            print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            winners.append(home)

        labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                        1/home_win_prob, 
                                                        world_cup_rankings.loc[away, 'country_abrv'], 
                                                        1/(1-home_win_prob)))
        odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    print("\n")


# The winner is Belgium.
# 
# Now let's correct the round 16 results with the actuals

# In[11]:


next_round_wc = pd.DataFrame(data={'team': ['Uruguay', 'Portugal', 'France', 'Argentina', 'Brazil', 'Mexico', 'Belgium', 'Japan',
                                           'Spain', 'Russia', 'Croatia', 'Denmark', 'Sweden', 'Switzerland', 'Colombia', 'England'],
                                  'group': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']})

next_round_wc = next_round_wc.reset_index().set_index('team')
finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

labels = list()
odds = list()

for f in finals:
    print("Starting of the {}".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home, away), end='')
        row = create_feature(X_test.columns, world_cup_rankings, home, away)
        
        home_win_prob = model.predict_proba(row)[:,1][0]
        if model.predict_proba(row)[:,1] <= 0.5:
            print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            winners.append(away)
        else:
            print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            winners.append(home)

        labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                        1/home_win_prob, 
                                                        world_cup_rankings.loc[away, 'country_abrv'], 
                                                        1/(1-home_win_prob)))
        odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    print("\n")


# Winner in this case is "England".
# 
# Now let's correct the prediction with actuals of round 16.

# In[12]:


next_round_wc = pd.DataFrame(data={'team': ['Uruguay', 'France', 'Brazil', 'Belgium',
                                           'Russia', 'Croatia', 'Sweden', 'England']})

next_round_wc = next_round_wc.reset_index().set_index('team')
finals = ['quarterfinal', 'semifinal', 'final']

labels = list()
odds = list()

for f in finals:
    print("Starting of the {}".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home, away), end='')
        row = create_feature(X_test.columns, world_cup_rankings, home, away)
        home_win_prob = model.predict_proba(row)[:,1][0]
        if model.predict_proba(row)[:,1] <= 0.5:
            print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            winners.append(away)
        else:
            print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            winners.append(home)

        labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                        1/home_win_prob, 
                                                        world_cup_rankings.loc[away, 'country_abrv'], 
                                                        1/(1-home_win_prob)))
        odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    print("\n")


# Winner in this case is "England".
# 
# 
# Further work
# 
# 1. We will be integrating bookie data if available.
# 
# 2. We will integrate the player data if available.
