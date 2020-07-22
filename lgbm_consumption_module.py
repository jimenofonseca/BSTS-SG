"""
This script cleans the data
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sg
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import shap
from auxiliary import week_of_month, BlockingTimeSeriesSplit
from config import DATA_CONSUMPTION_PROCESSED_FILE, INTERVENTION_CALENDAR, \
    DATA_WEATHER_PROCESSED_FILE, BEST_FEATURES_FILE, ALL_NUMERICAL_FEATURES,ALL_CATEGORICAL_FEATURES, \
    DATA_VACATIONS_INTERVENTION_FILE, DATA_METADATA_PROCESSED_FILE, DATA_HOLIDAYS_PROCESSED_FILE, \
    DATA_ISO_CONSUMPTION_PROCESSED_FILE, DATA_ENTHALPY_GRADIENTS_PROCESSED_FILE, DATA_VACATIONS_FILE, \
    DATA_SOLAR_GAINS_PROCESSED_FILE, DYNAMIC_SPACE, EXPERIMENTS, CONTROL_GROUPS, BEST_PARAMETERS_FILE
from custom_scorer_module import scorer_quantile, scorer_rmse


def lgbm_regression_efecto_acumulado_con_linea_base_del_experimento(alpha,
                                                                    data_mean,
                                                                    get_best_parameters=False,
                                                                    get_best_features=False,
                                                                    use_best_features=False):
    # INITIALIZE NEW FIELDS
    numerical_features_list = ALL_NUMERICAL_FEATURES
    categorical_features_list = ALL_CATEGORICAL_FEATURES
    new_field = "GBM_consumption_kWh_" + alpha
    data_mean[new_field] = 0.0
    for experiment, control in zip(EXPERIMENTS, CONTROL_GROUPS):

        # GET DATA ABOUT PERIODS OF INTERVENTION OF THE EXPERIMENTAL PERIOD
        intervention_data = INTERVENTION_CALENDAR[experiment]
        pre_period = intervention_data[1]
        post_period = intervention_data[2]
        range_pre_intervention_period = pd.date_range(start=pre_period[0], end=pre_period[1], freq='D')
        range_post_intervention_period = pd.date_range(start=post_period[0], end=post_period[1], freq='D')

        if use_best_features:
            best_features = open_best_features()
            features = best_features[alpha][str(experiment)]
            categorical_features_list = [x for x in categorical_features_list if x in features]
            numerical_features_list = [x for x in numerical_features_list if x in features]

        # GET TRAINING AND VALIDATION SET
        X, y = get_training_validation_set(CONTROL_GROUPS,
                                           data_mean,
                                           range_pre_intervention_period,
                                           categorical_features_list,
                                           numerical_features_list,
                                           experiment,
                                           "CONSUMPTION_kWh",
                                           get_best_parameters,
                                           get_best_features)

        if get_best_parameters and get_best_features:
            best_parameters = open_best_parameters()
            best_features = open_best_features()
            best_parameters[alpha][str(experiment)], best_features[alpha][
                str(experiment)] = get_best_params_and_features(X, y, float(alpha))
            save_best_parameters(best_parameters)
            save_best_features(best_features)
        elif get_best_parameters:
            best_parameters = open_best_parameters()
            best_parameters[alpha][str(experiment)] = get_best_params(X, y, float(alpha))
            save_best_parameters(best_parameters)
        elif get_best_features:
            best_parameters = open_best_parameters()
            params = best_parameters[alpha][str(experiment)]
            params['alpha'] = float(alpha)
            trained_model = lgb.train(params,
                                      lgb.Dataset(X, y, categorical_feature=categorical_features_list)
                                      )
            explainer = shap.TreeExplainer(trained_model)
            shap_values = explainer.shap_values(X)
            shap_sum = np.abs(shap_values).mean(axis=0)
            best_features = open_best_features()
            best_features[alpha][str(experiment)] = [f for f, s in zip(X.columns.tolist(), shap_sum.tolist()) if s >= 0.005]
            save_best_features(best_features)

            importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
            importance_df.columns = ['column_name', 'shap_importance']
            importance_df = importance_df.sort_values('shap_importance', ascending=False)
            print(importance_df)
        else:
            print('training experiment {}, with alpha {}, with control {}'.format(experiment, alpha, control))
            best_parameters = open_best_parameters()
            params = best_parameters[alpha][str(experiment)]
            params['alpha'] = float(alpha)
            trained_model = lgb.train(params,
                                      lgb.Dataset(X, y, categorical_feature=categorical_features_list)
                                      )

            # explainer = shap.TreeExplainer(trained_model)
            # shap_values = explainer.shap_values(X)
            # shap_sum = np.abs(shap_values).mean(axis=0)
            # importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
            # importance_df.columns = ['column_name', 'shap_importance']
            # importance_df = importance_df.sort_values('shap_importance', ascending=False)
            # print(importance_df)
            # fold_importance_df = pd.DataFrame()
            # fold_importance_df["feature"] = X.columns
            # fold_importance_df["importance"] = trained_model.feature_importance()
            # fold_importance_df = fold_importance_df.sort_values(by="importance", ascending=False)
            # print(fold_importance_df)

            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            # shap.summary_plot(shap_values, X, plot_type="bar")

            data_mean = predict(new_field,
                                data_mean,
                                trained_model,
                                experiment,
                                numerical_features_list,
                                categorical_features_list,
                                range_post_intervention_period,
                                range_pre_intervention_period)
    print("SUCCESS!")
    return data_mean


def open_best_parameters():
    with open(BEST_PARAMETERS_FILE, 'r') as fp:
        data = json.load(fp)
    return data


def open_best_features():
    with open(BEST_FEATURES_FILE, 'r') as fp:
        data = json.load(fp)
    return data


def save_best_parameters(best_parameters):
    with open(BEST_PARAMETERS_FILE, 'w') as fp:
        json.dump(best_parameters, fp)


def save_best_features(best_parameters):
    with open(BEST_FEATURES_FILE, 'w') as fp:
        json.dump(best_parameters, fp)


def predict(new_field,
            data_mean,
            final_model,
            experiment,
            numerical_features_list,
            categorical_features_list,
            range_post_intervention_period,
            range_pre_intervention_period):
    # GET ALL FEATURES
    features = numerical_features_list + categorical_features_list

    # ADOPT CATEGORICAL FEATURES SO THEY WORK AS EXPECTED
    for c in categorical_features_list:
        data_mean[c] = data_mean[c].astype('category')

    # PREDICT FOR PREINTERVENTION
    data_pre = data_mean[(data_mean.timestamp.isin(range_pre_intervention_period)) &
                         (data_mean.EXPERIMENT == experiment)]
    data_mean.loc[data_mean['timestamp'].isin(range_pre_intervention_period) &
                  (data_mean.EXPERIMENT == experiment), new_field] = np.exp(final_model.predict(data_pre[features]))

    # PREDICT FOR POSTINTERVENTION
    data_post = data_mean[(data_mean.timestamp.isin(range_post_intervention_period)) &
                          (data_mean.EXPERIMENT == experiment)]
    data_mean.loc[data_mean['timestamp'].isin(range_post_intervention_period) &
                  (data_mean.EXPERIMENT == experiment), new_field] = np.exp(
        final_model.predict(data_post[features]))

    return data_mean


def get_training_validation_set(control,
                                data_mean,
                                range_pre_intervention_period,
                                categorical_features_list,
                                numerical_features_list,
                                experiment,
                                target_feature_name,
                                get_best_parameters,
                                get_best_features):
    df = data_mean.copy()
    # GET TRAINING DATA
    if get_best_parameters or get_best_features:
        array = df[df.timestamp.isin(range_pre_intervention_period) |
                   (df.INTERVENTION.isin(control)) & (data_mean.EXPERIMENT == experiment)].values
        columns = df.columns
        data_train = pd.DataFrame(array, index=array[:, 0], columns=columns)
        data_train = data_train.reset_index(drop=True)
    else:
        array = df[df.timestamp.isin(range_pre_intervention_period) |
                   (df.INTERVENTION.isin(control)) & (data_mean.EXPERIMENT == experiment)].values
        # array = df[df.timestamp.isin(range_pre_intervention_period) & (df.EXPERIMENT == experiment)].values
        columns = df.columns
        data_train = pd.DataFrame(array, index=array[:, 0], columns=columns)
        data_train = data_train.reset_index(drop=True)

    # TRANSFORM INTO APPROPIATE VALUES
    for c in categorical_features_list:
        data_train[c] = data_train[c].astype('category')
    for c in numerical_features_list + [target_feature_name]:
        data_train[c] = data_train[c].astype('float32')

    # TRANSFORM IT TO LOG1P domain
    data_train[target_feature_name] = np.log(data_train[target_feature_name].astype('float'))

    # SPPLIT FINALLY
    features_list = numerical_features_list + categorical_features_list
    X = data_train[features_list]
    y = data_train[target_feature_name]

    return X, y

def get_best_params(X, y, alpha):
    grid_params = DYNAMIC_SPACE
    scoring = make_scorer(scorer_rmse, greater_is_better=False)#, quantile=alpha)
    regressor = lgb.LGBMRegressor(n_jobs=1,
                                  metric='quantile',
                                  objective='quantile')
    grid = GridSearchCV(estimator=regressor,
                        param_grid=grid_params,
                        verbose=1,
                        cv=BlockingTimeSeriesSplit(n_splits=5),
                        n_jobs=-1,
                        scoring=scoring)
    grid.fit(X, y)
    return grid.best_params_


def get_best_params_and_features(X, y, alpha):
    scoring = make_scorer(scorer_rmse, greater_is_better=False)#, quantile=alpha)
    regressor = lgb.LGBMRegressor(n_jobs=1,
                                  metric='quantile',
                                  objective='quantile')

    # selecf best features and then pass to the grid search
    selector = RFECV(estimator=regressor,
                     step=1,
                     cv=BlockingTimeSeriesSplit(n_splits=5),
                     scoring=scoring)

    # pipeline = Pipeline([("selector", selector), ("regressor",regressor)])

    grid_params = {}
    for name, space in DYNAMIC_SPACE.items():
        grid_params["estimator__" + name] = space

    grid = GridSearchCV(estimator=selector,
                        param_grid=grid_params,
                        verbose=1,
                        cv=BlockingTimeSeriesSplit(n_splits=5),
                        n_jobs=-1,
                        scoring=scoring)
    grid.fit(X, y)
    features = [f for f, s in zip(X.columns, grid.best_estimator_.support_) if s]
    final_grid_params = {}
    for name, space in grid.best_params_.items():
        final_grid_params[name.split("__")[-1]] = space

    return final_grid_params, features


def data_preprocessing_interventions():
    real_consumption_df = pd.read_csv(DATA_CONSUMPTION_PROCESSED_FILE)
    real_consumption_df['timestamp'] = pd.to_datetime(real_consumption_df['timestamp'])
    vacations_data_df = pd.read_csv(DATA_VACATIONS_INTERVENTION_FILE)
    vacations_data_df['timestamp'] = pd.to_datetime(vacations_data_df['timestamp'])
    vacations_data_smapee_df = pd.read_csv(DATA_VACATIONS_FILE)
    vacations_data_smapee_df['timestamp'] = pd.to_datetime(vacations_data_smapee_df['timestamp'])
    weather_data_df = pd.read_csv(DATA_WEATHER_PROCESSED_FILE)
    weather_data_df['timestamp'] = pd.to_datetime(weather_data_df['timestamp'])
    metadata_df = pd.read_excel(DATA_METADATA_PROCESSED_FILE, sheets='SENSORS')[['smapee',
                                                                                 'ID_CEA',
                                                                                 'INTERVENTION',
                                                                                 'EXPERIMENT',
                                                                                 'GFA_m2',
                                                                                 'INCOME',
                                                                                 'BEDROOMS']]
    holidays_df = pd.read_csv(DATA_HOLIDAYS_PROCESSED_FILE)
    holidays_df['timestamp'] = pd.to_datetime(holidays_df['timestamp'])
    gradients_df = pd.read_csv(DATA_ENTHALPY_GRADIENTS_PROCESSED_FILE)
    gradients_df['timestamp'] = pd.to_datetime(gradients_df['timestamp'])
    solar_gains_df = pd.read_csv(DATA_SOLAR_GAINS_PROCESSED_FILE)
    solar_gains_df['timestamp'] = pd.to_datetime(solar_gains_df['timestamp'])
    iso_consumption_df = pd.read_csv(DATA_ISO_CONSUMPTION_PROCESSED_FILE)
    iso_consumption_df['timestamp'] = pd.to_datetime(iso_consumption_df['timestamp'])

    # merge all the fields
    df = real_consumption_df.merge(metadata_df, left_on=['smapee', 'INTERVENTION'], right_on=['smapee', 'INTERVENTION'])
    df = df.merge(vacations_data_smapee_df, left_on=['timestamp','smapee', 'INTERVENTION'], right_on=['timestamp','smapee', 'INTERVENTION'])
    df = df.merge(weather_data_df, left_on='timestamp', right_on='timestamp')
    df = df.merge(gradients_df, left_on='timestamp', right_on='timestamp')
    df = df.merge(solar_gains_df, left_on=['timestamp', 'ID_CEA'], right_on=['timestamp', 'ID_CEA'])
    df = df.merge(iso_consumption_df, left_on=['timestamp', 'ID_CEA'], right_on=['timestamp', 'ID_CEA'])

    #now that all fields are merged then calculate absolute values of DEG and ISO models
    ACH = 2.4
    density_kg_m3 = 1.225
    florr_height_m = 3
    df['ISO_kWh'] = df['GFA_m2'] * df['ISO_Whm2'] / 1000
    df['DEG_C_kWh'] = (df['GFA_m2'] * ACH * florr_height_m * density_kg_m3) / 3600 * (df['DEG_C_kJperKg'])
    df['DEG_DEHU_kWh'] = (df['GFA_m2'] * ACH * florr_height_m * density_kg_m3) / 3600 * (df['DEG_DEHUM_kJperKg'])
    df['SG_kWh'] = df['GFA_m2'] * df['SG_Whm2'] / 1000
    # now group per intervention
    data_mean1 = df.groupby(['timestamp', 'INTERVENTION'])[['GFA_m2',
                                                            'BEDROOMS',
                                                            'INCOME',
                                                            'EMPTY_DAY',
                                                            'FULL_DAY',
                                                            ]].sum()
    data_mean2 = df.groupby(['timestamp', 'INTERVENTION'])[['CONSUMPTION_kWh',
                                                            'GFA_m2',
                                                            'BEDROOMS',
                                                            'INCOME',
                                                            'ISO_kWh',
                                                            'SG_kWh',
                                                            'DEG_C_kWh',
                                                            'DEG_DEHU_kWh',
                                                            'EMPTY_DAY',
                                                            'FULL_DAY',
                                                            'EXPERIMENT']].mean()
    data_mean = data_mean1.merge(data_mean2, left_index=True, right_index=True,suffixes=('_SUM', '_MEAN') )
    data_mean = data_mean.rename(columns={"EXPERIMENT_MEAN": "EXPERIMENT"})
    data_mean = data_mean.reset_index()
    data_mean['year'] = np.array(data_mean['timestamp'].dt.year, dtype=np.uint16)
    data_mean['month'] = np.array(data_mean['timestamp'].dt.month, dtype=np.uint8) - 1
    data_mean['dayofweek'] = np.array(data_mean['timestamp'].dt.dayofweek, dtype=np.uint8)
    data_mean['dayofyear'] = np.array(data_mean['timestamp'].dt.dayofyear, dtype=np.uint16) - 1
    data_mean['weekofyear'] = np.array(data_mean['timestamp'].dt.weekofyear, dtype=np.uint8) - 1
    data_mean['weekday'] = data_mean['dayofweek'].apply(lambda x: 1 if 0 <= x < 5 else 0)
    data_mean['calendar_wom'] = data_mean['timestamp'].apply(week_of_month)

    # integrate vacation data and holidays data #common for all
    data_mean = data_mean.merge(vacations_data_df, left_on=['INTERVENTION', 'timestamp'],
                                right_on=['INTERVENTION', 'timestamp'])
    data_mean = data_mean.merge(holidays_df, left_on='timestamp', right_on='timestamp')

    data_mean = add_sg(data_mean, 'INTERVENTION', 29, diff=1)
    data_mean = add_sg(data_mean, 'INTERVENTION', 89, diff=1)
    return data_mean

def add_sg(df, group, lag, diff):
    p = 3
    df['ISO_kWh_smooth_'+str(lag)] = 0.0
    df['SG_kWh_smooth_'+str(lag)] = 0.0
    df['DEG_C_kWh_smooth_'+str(lag)] = 0.0
    for si in df[group].unique():
        index = df[group] == si
        df.loc[index, 'ISO_kWh_smooth_'+str(lag)] = sg(df[index].ISO_kWh, lag, p)
        df.loc[index, 'SG_kWh_smooth_'+str(lag)] = sg(df[index].SG_kWh, lag, p)
        df.loc[index, 'DEG_C_kWh_smooth_'+str(lag)] = sg(df[index].DEG_C_kWh, lag, p)
    return df

