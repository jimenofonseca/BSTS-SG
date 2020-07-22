"""
This script cleans the data
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from custom_scorer_module import scorer_rmse
from sklearn.metrics import make_scorer
from auxiliary import GrupedTimeseriesKFold
from config import SEED

def imputer(df: pd.DataFrame,
            timestamp_feature_name: str,
            target_feature_name: str,
            numerical_features_list: list,
            categorical_features_list: list,
            id_column: str,
            window: int = 7,
            get_best_parameters: bool = False,
            params: dict = {'learning_rate': 0.01,'num_leaves': 16,'min_data_in_leaf': 1000,'num_iterations': 10000,'objective': 'rmse','metric': 'rmse'},
            groupby:str='smapee'
            ):
    """

    :param data:
    :param target_column:
    :param window:
    :param id_column:
    :return:
    """
    data = df.copy() #because it gets changed and lgbm breaks
    # rolling
    data['rolling_back'] = data.groupby(by=id_column)[target_feature_name] \
        .rolling(window=window, min_periods=1).mean().interpolate().values

    # reversed rolling
    data['rolling_forw'] = data.iloc[::-1].groupby(by=id_column)[target_feature_name] \
        .rolling(window=window, min_periods=1).mean().interpolate().values

    # rolling mean for same hour of the week
    data['rolling_back_h'] = data.groupby(by=[id_column, 'dayofweek'])[target_feature_name] \
        .rolling(window=3, min_periods=1).mean().interpolate().values

    data['rolling_back_h_f'] = data.iloc[::-1].groupby(by=[id_column, 'dayofweek'])[target_feature_name] \
        .rolling(window=3, min_periods=1).mean().interpolate().values

    tr_idx, val_idx = ~data[target_feature_name].isnull(), data[target_feature_name].isnull()

    numerical_features_list = numerical_features_list + ['rolling_back',
                                                         'rolling_forw',
                                                         'rolling_back_h',
                                                         'rolling_back_h_f']
    features_list = numerical_features_list + categorical_features_list

    if get_best_parameters:
        # train in the log domain
        data[target_feature_name] = np.log(1 + data[target_feature_name])
        X = data.loc[tr_idx, features_list]
        X = X.reset_index(drop=True)
        y = data.loc[tr_idx, target_feature_name]
        y = y.reset_index(drop=True)

        grid_params = {'learning_rate': [0.001, 0.01, 0.1],
                       'num_leaves': [4, 16, 32, 64],
                       'max_depth': [-1],
                       'num_iterations': [10000],
                       'min_data_in_leaf': [20, 100, 200, 500],
                       'boosting': ['gbdt']}

        mdl = lgb.LGBMRegressor(n_jobs=1,
                                metric='rmse',
                                objective='rmse',
                                seed=SEED)

        grid = GridSearchCV(mdl,
                            grid_params,
                            verbose=1,
                            cv=GrupedTimeseriesKFold(groupby=groupby),
                            n_jobs=-1,
                            scoring=make_scorer(scorer_rmse, greater_is_better=False))
        # Run the grid
        grid.fit(X, y)

        # Print the best parameters found
        print(grid.best_params_)
        print(grid.best_score_)

    else:
        evals_result = {}  # to record eval results for plotting

        # train in the log domain
        data[target_feature_name] = np.log(1 + data[target_feature_name])
        X = data.loc[tr_idx, features_list]
        y = data.loc[tr_idx, target_feature_name]

        lgb_train = lgb.Dataset(X, y, categorical_feature=categorical_features_list)
        lgb_eval = lgb.Dataset(X, y, categorical_feature=categorical_features_list)

        reg = lgb.train(params,
                        lgb_train,
                        valid_sets=(lgb_train, lgb_eval),
                        evals_result=evals_result,
                        early_stopping_rounds=5000,
                        verbose_eval=10000)

        data[f'{target_feature_name}_imputed'] = np.nan
        data.loc[val_idx, f'{target_feature_name}_imputed'] = reg.predict(data.loc[val_idx, features_list])
        data.loc[val_idx, f'{target_feature_name}'] = data.loc[val_idx, f'{target_feature_name}_imputed'].values

        # return to the real domain
        data[target_feature_name] = np.exp(data[target_feature_name]) - 1
        data[f'{target_feature_name}_imputed'] = np.exp(data[f'{target_feature_name}_imputed']) - 1

        #also get some idea of how it looks for all the data
        data[f'{target_feature_name}_all_imputed'] = np.exp(reg.predict(data[features_list]))-1

        # check from what features_list our imputer learned the most
        lgb.plot_importance(reg, title=f'feature importance for "{target_feature_name}"')
        lgb.plot_metric(evals_result, metric='rmse')

    return data
