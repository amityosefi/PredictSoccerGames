import sqlite3
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

df_columns = ['home_team_wins', 'home_team_draws', 'home_team_losses', 'away_team_wins', 'away_team_draws', 'away_team_losses',
              'home_team_goals', 'home_opponent_goals', 'away_opponent_goals', 'away_team_goals', 'head_to_head',
              'home_players_avg', 'away_players_avg', 'betting_home', 'betting_away', 'betting_draw', 'label']


def game_result(team1, team2):
    """
    calculates game winner
    :param team1: scores of team 1
    :param team2: scores of team 2
    :return: 1 = team1 wins, 0 = tie, -1 = team2 wins
    """
    if team1 > team2:
        return 1
    elif team1 < team2:
        return -1
    return 0


def home_team_statistics(df_match, team_id, date):
    df_match = df_match[df_match.date < date]
    df_match = df_match[df_match.home_team_api_id == team_id]

    stats = [0, 0, 0, 0, 0]  # [wins, ties, losses, goals, opponentGoals]
    for index, match in df_match.iterrows():
        if game_result(match['home_team_goal'], match['away_team_goal']) == 1:
            stats[0] += 1
        elif game_result(match['home_team_goal'], match['away_team_goal']) == 0:
            stats[1] += 1
        else:
            stats[2] += 1
        stats[3] += match['home_team_goal']
        stats[4] += match['away_team_goal']

    return stats


def away_team_statistics(df_match, team_id, date):
    df_match = df_match[df_match.date < date]
    df_match = df_match[df_match.away_team_api_id == team_id]

    stats = [0, 0, 0, 0, 0]  # [wins, ties, losses, goals, opponentGoals]
    for index, match in df_match.iterrows():
        if game_result(match['home_team_goal'], match['away_team_goal']) == -1:
            stats[0] += 1
        elif game_result(match['home_team_goal'], match['away_team_goal']) == 0:
            stats[1] += 1
        else:
            stats[2] += 1
        stats[3] += match['away_team_goal']
        stats[4] += match['home_team_goal']

    return stats


def update_results(team_id, team_stats, stats):
    if team_id not in team_stats:
        team_stats[team_id] = stats
        return [0, 0, 0, 0, 0]

    results = team_stats[team_id]
    team_stats[team_id][0] += stats[0]
    team_stats[team_id][1] += stats[1]
    team_stats[team_id][2] += stats[2]
    team_stats[team_id][3] += stats[3]
    team_stats[team_id][4] += stats[4]
    return results


def get_wins(stats):
    return stats[0]


def get_draws(stats):
    return stats[1]


def get_losses(stats):
    return stats[2]


def get_goals(stats):
    return stats[3]


def get_opponent_goals(stats):
    return stats[4]


def head_to_head(df_match, team1, team2, date):
    """
    counts how many games 2 teams won in head to head game.
    :return: number of wins in head to head games between team 1 and 2.
    positive numbers are team 1 wins, negative are team 2 wins
    """
    results = 0
    df_match = df_match[df_match.date < date]
    team1_home = df_match[df_match.home_team_api_id == team1]
    team1_home = team1_home[team1_home.away_team_api_id == team2]
    for index, match in team1_home.iterrows():
        results += game_result(match['home_team_goal'], match['away_team_goal'])

    team2_home = df_match[df_match.home_team_api_id == team2]
    team2_home = team2_home[team2_home.away_team_api_id == team1]
    for index, match in team2_home.iterrows():
        results += game_result(match['away_team_goal'], match['home_team_goal'])

    return results


def get_overall_rating(attributes, player_id, date):
    player_data = attributes[attributes.player_api_id == player_id]
    return player_data[player_data.date < date].iloc[0]['overall_rating']



def normalize_goals(games, statistics):
    """
    normalize goals by dividing by total number of games
    :return: normalized goals
    """
    if games != 0:
        goals = get_goals(statistics) / games
        opponent_goals = get_opponent_goals(statistics) / games
        return goals, opponent_goals

    return 0, 0


def normalize_games(games, wins, draws, loses):
    """
    normalize scores by dividing by total number of games
    :return: normalized scores
    """
    if games != 0:
        wins = wins / games
        draws = draws / games
        loses = loses / games
        return wins, draws, loses

    return 0, 0, 0


def get_bets(match, avg, B365, BW):
    if match[B365] != (-1) and match[BW] != (-1):
        return (match[B365] + match[BW]) / 2
    elif match[B365] != (-1) and match[BW] == (-1):
        return match[B365]
    elif match[BW] != (-1) and match[B365] == (-1):
        return match[BW]
    else:  # both are -1, empty data
        return avg


def init_df(match_data, player_data, bet_home_avg, bet_away_avg, bet_draw_avg):

    df = pd.DataFrame(columns=df_columns)

    for index, match in match_data.iterrows():

        date = match['date']
        home_id = match['home_team_api_id']
        home_statistics = home_team_statistics(match_data, home_id, date)
        home_wins = get_wins(home_statistics)
        home_losses = get_losses(home_statistics)
        home_draws = get_draws(home_statistics)
        home_games = home_wins + home_draws + home_losses
        home_wins, home_draw, home_losses = normalize_games(home_games, home_wins, home_draws, home_losses)
        home_goals, home_opponent_goals = normalize_goals(home_games, home_statistics)

        away_id = match['away_team_api_id']
        away_statistics = away_team_statistics(match_data, away_id, date)
        away_wins = get_wins(away_statistics)
        away_draws = get_draws(away_statistics)
        away_losses = get_losses(away_statistics)
        away_games = away_wins + away_draws + away_losses
        away_goals, away_opponent_goals = normalize_goals(away_games, away_statistics)
        away_wins, away_draw, away_losses = normalize_games(away_games, away_wins, away_draws, away_losses)

        head2head = head_to_head(match_data, home_id, away_id, date)

        bet_home = get_bets(match, bet_home_avg, 'B365H', 'BWH')
        bet_draw = get_bets(match, bet_draw_avg, 'B365D', 'BWD')
        bet_away = get_bets(match, bet_away_avg, 'B365A', 'BWA')

        label = game_result(match['home_team_goal'], match['away_team_goal'])

        player_overall_avg = [0, 0]  # [home_avg, away_avg]
        count_home, count_away = 0, 0
        for i in range(1, 12):
            if match["home_player_" + str(i)] != (-1):
                player_overall = get_overall_rating(player_data, match["home_player_" + str(i)], date)
                player_overall_avg[0] += player_overall
                count_home += 1
            if match["away_player_" + str(i)] != (-1):
                player_overall = get_overall_rating(player_data, match["away_player_" + str(i)], date)
                player_overall_avg[1] += player_overall
                count_away += 1

        # if there are at least 4 players with data (not -1) get avg - divide by number of players
        if count_away >= 4 and count_home >= 4:
            player_overall_avg[0] /= count_home
            player_overall_avg[1] /= count_away
        else:
            # otherwise - not enough data on players, we will skip this row
            continue

        data_dict = {'home_team_wins': home_wins, 'home_team_draws': home_draw,
                     'home_team_losses': home_losses,
                     'away_team_wins': away_wins, 'away_team_draws': away_draw,
                     'away_team_losses': away_losses,
                     'home_team_goals': home_goals, 'home_opponent_goals': home_opponent_goals,
                     'away_opponent_goals': away_opponent_goals, 'away_team_goals': away_goals,
                     'head_to_head': head2head,
                     'home_players_avg': player_overall_avg[0], 'away_players_avg': player_overall_avg[1],
                     'betting_home': bet_home, 'betting_away': bet_away,
                     'betting_draw': bet_draw, 'label': label}

        df = df.append(data_dict, ignore_index=True)

    return df

def get_mean(data, B365, BWH):
    """ get average of each column - and get average between them - this is for when betting data
     will have na - we will replace it with this avg
     """
    return (data[B365].mean() + data[BWH].mean()) / 2

def get_classification(model_str, test, pred):
    return model_str + "\n" + classification_report(test, pred, zero_division=1)

def most_frequent(lst):
    counter = 0
    num = lst[0]
    for j in lst:
        curr_frequency = lst.count(j)
        if curr_frequency > counter:
            counter = curr_frequency
            num = j
    return num

if __name__ == '__main__':
    con = sqlite3.connect("database.sqlite")
    match_select = 'season, home_team_api_id, away_team_api_id, date, home_team_goal, away_team_goal,' \
                   'home_player_1,home_player_2,home_player_3,home_player_4,home_player_5,home_player_6,home_player_7,' \
                   'home_player_8,home_player_9,home_player_10,home_player_11,' \
                   'away_player_1,away_player_2,away_player_3,away_player_4,away_player_5,away_player_6,away_player_7,' \
                   'away_player_8,away_player_9,away_player_10,away_player_11,B365H,B365D,B365A,BWH,BWD,BWA'

    match_test_data = pd.read_sql('SELECT ' + match_select + ' FROM Match where season="2015/2016";', con)
    match_train_data = pd.read_sql('SELECT ' + match_select + ' FROM Match where season!="2015/2016";', con)

    # averages for betting columns (for later if row has null)
    bet_home_avg_train = get_mean(match_train_data, 'B365H', 'BWH')
    bet_away_avg_train = get_mean(match_train_data, 'B365A', 'BWA')
    bet_draw_avg_train = get_mean(match_train_data, 'B365D', 'BWD')

    bet_home_avg_test = get_mean(match_test_data, 'B365H', 'BWH')
    bet_away_avg_test = get_mean(match_test_data, 'B365A', 'BWA')
    bet_draw_avg_test = get_mean(match_test_data, 'B365D', 'BWD')

    # replace na with -1
    match_test_data = match_test_data.fillna(-1)
    match_train_data = match_train_data.fillna(-1)

    df_player_attributes = pd.read_sql_query("SELECT player_api_id, date, overall_rating from Player_Attributes", con)
    df_player_attributes.dropna(axis=0, inplace=True)

    train_df = init_df(match_train_data, df_player_attributes, bet_home_avg_train,
                       bet_away_avg_train, bet_draw_avg_train)
    test_df = init_df(match_test_data, df_player_attributes, bet_home_avg_test, bet_away_avg_test,
                      bet_draw_avg_test)

    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")

    x_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    x_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    # models
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(x_train, y_train)
    y_pred_LDA = LDA.predict(x_test)
    print('LDA accuracy: ', metrics.accuracy_score(y_test, y_pred_LDA))

    ADA_model = AdaBoostClassifier(n_estimators=200, random_state=2)
    model = ADA_model.fit(x_train, y_train)
    y_pred_ADA = model.predict(x_test)
    print("AdaBoostClassifier Accuracy:", metrics.accuracy_score(y_test, y_pred_ADA))

    LR_model = LogisticRegression(solver='liblinear', multi_class='ovr')
    LR_model.fit(x_train, y_train)
    y_pred_LR = LR_model.predict(x_test)
    print('LogisticRegression accuracy: ', metrics.accuracy_score(y_test, y_pred_LR))

    MLP_model = MLPClassifier(hidden_layer_sizes=(15,), random_state=2, max_iter=1000)
    MLP_model.fit(x_train, y_train)
    y_pred_MLP = MLP_model.predict(x_test)
    print("MLP Accuracy:", metrics.accuracy_score(y_test, y_pred_MLP))

    # combining the result vectors to one result vector - the best one!
    # by putting the most common value (-1/0/1) of the 3 vectors in the result vector
    res = []
    for i in range(len(y_test)):
        res.append(most_frequent([y_pred_ADA[i], y_pred_LDA[i], y_pred_LR[i]]))
    y_pred_union = np.array(res)
    print("Union Accuracy:", metrics.accuracy_score(y_test, y_pred_union))

    print(get_classification("LDA data: ", y_test, y_pred_LDA))
    print(get_classification("AdaBoostClassifier data: ", y_test, y_pred_ADA))
    print(get_classification("LogisticRegression data: ", y_test, y_pred_LR))
    print(get_classification("MLP data: ", y_test, y_pred_MLP))
    print(get_classification("Union Data: ", y_test, y_pred_union))

    con.close()
