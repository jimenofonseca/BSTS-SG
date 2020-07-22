import calendar
import datetime

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from causalimpact import CausalImpact
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import INTERVENTION_CALENDAR


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


class GrupedTimeseriesKFold():

    def __init__(self, n_splits=5, groupby='smapee'):
        self.n_splits = n_splits
        self.groupby = groupby

    def split(self, X, y=None, groups=None, ):
        groups = X.groupby(self.groupby).groups
        split_trains1 = []
        split_tests1 = []
        split_trains2 = []
        split_tests2 = []
        split_trains3 = []
        split_tests3 = []
        split_trains4 = []
        split_tests4 = []
        split_trains5 = []
        split_tests5 = []
        for group, indexes in groups.items():
            tscv = TimeSeriesSplit(n_splits=5)
            counter = 0
            for train_index, test_index in tscv.split(indexes):
                if counter == 0:
                    split_trains1.extend(indexes[train_index].values)
                    split_tests1.extend(indexes[test_index].values)
                elif counter == 1:
                    split_trains2.extend(indexes[train_index].values)
                    split_tests2.extend(indexes[test_index].values)
                elif counter == 2:
                    split_trains3.extend(indexes[train_index].values)
                    split_tests3.extend(indexes[test_index].values)
                elif counter == 3:
                    split_trains4.extend(indexes[train_index].values)
                    split_tests4.extend(indexes[test_index].values)
                elif counter == 4:
                    split_trains5.extend(indexes[train_index].values)
                    split_tests5.extend(indexes[test_index].values)
                else:
                    print("ERROR")
                counter += 1
        cv = [(split_trains1, split_tests1),
              (split_trains2, split_tests2),
              (split_trains3, split_tests3),
              (split_trains4, split_tests4),
              (split_trains5, split_tests5)]
        for rxm, tx in cv:
            yield (np.array(rxm), np.array(tx))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def week_of_month(tgtdate):
    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days // 7 + 1


def graph_check_gbm_timeseries(data_mean, X_names, y_limits):
    visual = data_mean.set_index('timestamp')
    intervention_list = visual['INTERVENTION'].unique()
    # Set up the matplotlib figure
    f, axes = plt.subplots(4, 3, figsize=(15, 9))
    sns.despine(left=True)
    fields_to_plot = ['CONSUMPTION_kWh'] + X_names
    for i, intervention in enumerate(intervention_list):
        plot = visual[visual['INTERVENTION'] == intervention]
        if list(plot['INTERVENTION'].values[0])[-1] == 'L':
            row = 0
            column = int(list(plot['INTERVENTION'].values[0])[0]) - 1
        else:
            row = int(list(plot['INTERVENTION'].values[0])[-1])
            column = int(list(plot['INTERVENTION'].values[0])[0]) - 1

        if intervention == '2T4':
            row = 1
            column = 1
        elif intervention == '2T5':
            row = 2
            column = 1

        ax = axes[row, column]
        ax.set_ylim(y_limits[0], y_limits[1])
        plot[fields_to_plot].plot(ax=ax, title=intervention)
    font = {'family': 'Arial',
            'size': 10}
    matplotlib.rc('font', **font)


def graph_check_gbm_dist(data_mean, X_names):
    visual = data_mean.set_index('timestamp')
    visual['observation'] = np.log1p(visual.CONSUMPTION_kWh)
    intervention_list = visual['INTERVENTION'].unique()
    # Set up the matplotlib figure
    f, axes = plt.subplots(4, 3, figsize=(10, 9))
    sns.despine(left=True)
    for i, intervention in enumerate(intervention_list):
        plot = visual[visual['INTERVENTION'] == intervention]

        if list(plot['INTERVENTION'].values[0])[-1] == 'L':
            row = 0
            column = int(list(plot['INTERVENTION'].values[0])[0]) - 1
        else:
            row = int(list(plot['INTERVENTION'].values[0])[-1])
            column = int(list(plot['INTERVENTION'].values[0])[0]) - 1

            if intervention == '2T4':
                row = 1
                column = 1
            elif intervention == '2T5':
                row = 2
                column = 1

        ax = axes[row, column]

        # plot observation
        sns.distplot(plot.observation, hist=False, ax=ax, kde_kws={"label": intervention})

        # plit predction
        for field in X_names:
            value = np.log1p(plot[field])
            legend = round(np.sqrt(mean_squared_error(value, plot.observation)), 4)
            sns.distplot(value, ax=ax, hist=False, kde_kws={"label": field.split("_")[-1] + " " + str(legend)})


def prepare_data_synthetic_bsts(data, INTERVENTION, X_names):
    # let's get the data for the first experiment of sensors in VIEW
    data_selection = data[data['INTERVENTION'] == INTERVENTION]
    dict_info = {'y': data_selection['CONSUMPTION_kWh']}
    for i, field in enumerate(X_names):
        dict_info["x" + str(i)] = data_selection[field]
    dat_final = pd.DataFrame(dict_info)
    return dat_final


def prepare_data_control_bsts(data, INTERVENTION, X_names):
    # let's get the data for the first experiment of sensors in VIEW
    # let's get the data for the first experiment of sensors in VIEW
    df = data.copy()
    data_selection = df[df['INTERVENTION'] == INTERVENTION]
    data_control = df[df['INTERVENTION'] == list(INTERVENTION)[0] + 'CONTROL']
    data_mean = data_selection.merge(data_control, left_index=True, right_index=True, suffixes=('', '_y'))
    dict_info = {'y':data_mean['CONSUMPTION_kWh'],
                 'x1':data_mean['CONSUMPTION_kWh_y']}
    data = pd.DataFrame(dict_info)
    return data, data_mean.index


def graph_check_cumulative_bsts(data_mean, X_names):
    visual = data_mean.set_index('timestamp')
    intervention_list = visual['INTERVENTION'].unique()
    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 3, figsize=(15, 9))
    dataframe2 = pd.DataFrame()
    for i, intervention in enumerate(intervention_list):
        if list(intervention)[-1] == 'L':
            x = 1  # do nothing
        else:
            experiment = int(list(intervention)[0])
            intervention_data = INTERVENTION_CALENDAR[experiment]
            pre_period = intervention_data[1]
            post_period = intervention_data[2]
            end_intervention_date = intervention_data[3]

            # get position for the plot
            row = int(list(intervention)[-1]) - 1
            column = int(list(intervention)[0]) - 1

            if intervention == '2T4':
                row = 0
                column = 1
            elif intervention == '2T5':
                row = 1
                column = 1

            ax = axes[row, column]
            data = prepare_data_synthetic_bsts(data_mean.set_index('timestamp'), intervention, X_names)
            ci = CausalImpact(data, pre_period, post_period, prior_level_sd=None, standarize=True)
            ax = ci.plot(figsize=(5, 3), end_intervention_date=end_intervention_date, panels=['cumulative'],
                         add_axes=ax)
            ax.set_title(intervention)

            # get data
            table = ci.summary_data
            pi_value = ci.p_value
            effect = str(round(table.loc['rel_effect', 'average'] * 100, 2)) + '/n' + '[' + str(
                round(table.loc['rel_effect_lower', 'average'] * 100, 2)) + ',' + str(
                round(table.loc['rel_effect_upper', 'average'] * 100, 2)) + ']'
            table_df = pd.DataFrame({'id': [intervention], 'effect': [effect], 'p_value': [pi_value]})
            dataframe2 = dataframe2.append(table_df, ignore_index=True)
    print(dataframe2)
    plt.show()
    font = {'family': 'Arial',
            'size': 10}
    matplotlib.rc('font', **font)


def graph_check_all_bsts(data_mean, intervention, X_names, title):
    experiment = int(list(intervention)[0])
    intervention_data = INTERVENTION_CALENDAR[experiment]
    pre_period = intervention_data[1]
    post_period = intervention_data[2]
    end_intervention_date = intervention_data[3]
    data = prepare_data_synthetic_bsts(data_mean.set_index('timestamp'), intervention, X_names)
    x = data.copy()
    x = x.rename(columns={'y':'Observation', 'x0':'Bayes. Synth. Control Group'})
    ci = CausalImpact(data, pre_period, post_period, prior_level_sd=None, standarize=True)
    font = {'family': 'Arial',
            'size': 18}
    ci.plot(figsize=(7, 9), end_intervention_date=end_intervention_date, title=title)
    matplotlib.rc('font', **font)
    return ci, x

def graph_check_all_bsts_control(data_mean, intervention, X_names, title):
    experiment = int(list(intervention)[0])
    intervention_data = INTERVENTION_CALENDAR[experiment]
    pre_period = intervention_data[1]
    post_period = intervention_data[2]
    end_intervention_date = intervention_data[3]
    data, time = prepare_data_control_bsts(data_mean.set_index('timestamp'), intervention, X_names)
    x = data.copy()
    x = x.rename(columns={'y':'Observation', 'x1':'Random. Control Group'})
    ci = CausalImpact(data, pre_period, post_period, prior_level_sd=None, standarize=True)
    font = {'family': 'Arial',
            'size': 18}
    ci.plot(figsize=(7, 9), end_intervention_date=end_intervention_date, title=title)
    matplotlib.rc('font', **font)
    return ci, x


def graph_check_all_bsts_table(data_mean, intervention, X_names, n):
    experiment = int(list(intervention)[0])
    intervention_data = INTERVENTION_CALENDAR[experiment]
    pre_period = intervention_data[1]
    post_period = intervention_data[2]
    data = prepare_data_synthetic_bsts(data_mean.set_index('timestamp'), intervention, X_names)
    ci = CausalImpact(data, pre_period, post_period, prior_level_sd=None, standarize=True)
    table = ci.summary_data
    # Min_avg = ["-",
    #            round(table.loc['predicted_lower', 'average'],2),
    #            round(table.loc['abs_effect_lower','average'],2),
    #            round(table.loc['rel_effect_lower','average']*100,2)]
    # M_avg = [round(table.loc['actual','average'],2),
    #          round(table.loc['predicted', 'average'],2),
    #          round(table.loc['abs_effect','average'],2),
    #          round(table.loc['rel_effect','average']*100,2)]
    # SD_avg = ["-",
    #          round(table.loc['predicted_sd', 'average'],2),
    #          round(table.loc['abs_effect_sd','average'],2),
    #          round(table.loc['rel_effect_sd','average']*100,2)]
    # Max_avg = ["-",
    #            round(table.loc['predicted_upper','average'],2),
    #            round(table.loc['abs_effect_upper','average'],2),
    #            round(table.loc['rel_effect_upper','average']*100,2)]
    #
    # Min_cum = ["-",
    #            round(table.loc['predicted_lower', 'cumulative'],2),
    #            round(table.loc['abs_effect_lower','cumulative'],2),
    #            round(table.loc['rel_effect_lower','cumulative']*100,2)]
    # M_cum = [round(table.loc['actual','cumulative'],2),
    #          round(table.loc['predicted', 'cumulative'],2),
    #          round(table.loc['abs_effect','cumulative'],2),
    #          round(table.loc['rel_effect','cumulative']*100,2)]
    # SD_cum = ["-",
    #          round(table.loc['predicted_sd', 'cumulative'],2),
    #          round(table.loc['abs_effect_sd','cumulative'],2),
    #          round(table.loc['rel_effect_sd','cumulative']*100,2)]
    # Max_acum = ["-",
    #            round(table.loc['predicted_upper','cumulative'],2),
    #            round(table.loc['abs_effect_upper','cumulative'],2),
    #            round(table.loc['rel_effect_upper','cumulative']*100,2)]
    #
    # data = pd.DataFrame({"Treatment": ["Treatement "+intervention+ " (n="+str(n)+")", "","",""],
    #               "Data": ["Observation (kWh)",
    #                        "Counterfactual (kWh)",
    #                        "Absolute effect (kWh)",
    #                        "Relative effect (%)"],
    #               "Min_avg":Min_avg,
    #               "M_avg":M_avg,
    #               "SD_avg":SD_avg,
    #               "Max_avg":Max_avg,
    #               "Min_cum": Min_cum,
    #               "M_cum": M_cum,
    #               "SD_cum": SD_cum,
    #               "Max_acum": Max_acum,
    #               })
    data = pd.DataFrame({"Treatment": ["Treatement " + intervention + "(n=" + str(n) + ")"],
                         "Observation": str(round(table.loc['actual','average'],2))+" kWh",
                         "Counterfactual\n(s.d.)": put_together_two(round(table.loc['predicted', 'average'],2), round(table.loc['predicted_sd', 'average'],2)),
                         "Absolute effect\n[95% c.i.]": put_together_three(round(table.loc['abs_effect', 'average'],2), round(table.loc['abs_effect_lower', 'average'],2), round(table.loc['abs_effect_upper', 'average'],2)),
                         "Percentage Change\n[95% c.i.]": put_together_three_perc(round(table.loc['rel_effect', 'average']*100,2), round(table.loc['rel_effect_lower', 'average']*100,2), round(table.loc['rel_effect_upper', 'average']*100,2)),
                         "Observation ": str(round(table.loc['actual', 'cumulative'], 2))+" kWh",
                         "Counterfactual\n(s.d.) ": put_together_two(round(table.loc['predicted', 'cumulative'], 2),
                                                                  round(table.loc['predicted_sd', 'cumulative'], 2)),
                         "Absolute effect\n[95% c.i.] ": put_together_three(round(table.loc['abs_effect', 'cumulative'], 2),
                                                                       round(table.loc['abs_effect_lower', 'cumulative'],
                                                                             2),
                                                                       round(table.loc['abs_effect_upper', 'cumulative'],
                                                                             2)),
                         "Percentage Change\n[95% c.i.] ": put_together_three_perc(
                             round(table.loc['rel_effect', 'cumulative'] * 100, 2),
                             round(table.loc['rel_effect_lower', 'cumulative'] * 100, 2),
                             round(table.loc['rel_effect_upper', 'cumulative'] * 100, 2)),
                         })

    return data


def graph_point_effects(data_mean, intervention, X_names):
    experiment = int(list(intervention)[0])
    intervention_data = INTERVENTION_CALENDAR[experiment]
    pre_period = intervention_data[1]
    post_period = intervention_data[2]
    data = prepare_data_synthetic_bsts(data_mean.set_index('timestamp'), intervention, X_names)
    ci = CausalImpact(data, pre_period, post_period, prior_level_sd=None, standarize=True)
    point_effects = ci.inferences['point_effects'].values
    return point_effects



def put_together_two(a,b):
    return str(a)+" kWh\n("+str(b)+")"

def put_together_three(a,b,c):
    return str(a)+" kWh\n["+str(b)+","+str(c)+"]"

def put_together_three_perc(a,b,c):
    return str(a)+"%\n["+str(b)+","+str(c)+"]"