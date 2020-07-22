
def add_sg(df, group):
    p = 2
    df['ISO_kWh_MEAN_roll_3'] = 0.0
    df['ISO_kWh_MEAN_roll_7'] = 0.0
    df['ISO_kWh_MEAN_roll_29'] = 0.0
    df['DEG_kWh_MEAN_roll_3'] = 0.0
    df['DEG_kWh_MEAN_roll_7'] = 0.0
    df['DEG_kWh_MEAN_roll_29'] = 0.0
    for si in df[group].unique():
        index = df[group] == si
        df.loc[index, 'ISO_kWh_MEAN_roll_3'] = sg(df[index].ISO_kWh, 3, p)
        df.loc[index, 'ISO_kWh_MEAN_roll_7'] = sg(df[index].ISO_kWh, 7, p)
        df.loc[index, 'ISO_kWh_MEAN_roll_29'] = sg(df[index].ISO_kWh, 29, p)
        df.loc[index, 'DEG_kWh_MEAN_roll_3'] = sg(df[index].DEG_kWh, 3, p)
        df.loc[index, 'DEG_kWh_MEAN_roll_7'] = sg(df[index].DEG_kWh, 7, p)
        df.loc[index, 'DEG_kWh_MEAN_roll_29'] = sg(df[index].DEG_kWh, 29, p)

        df.loc[index, 'ISO_kWh_lag_3'] = sg(df[index].ISO_kWh, 3, p)
        df.loc[index, 'ISO_kWh_lag_7'] = sg(df[index].ISO_kWh, 7, p)
        df.loc[index, 'ISO_kWh_lag_29'] = sg(df[index].ISO_kWh, 29, p)
        df.loc[index, 'DEG_kWh_lag_3'] = sg(df[index].DEG_kWh, 3, p)
        df.loc[index, 'DEG_kWh_lag_7'] = sg(df[index].DEG_kWh, 7, p)
        df.loc[index, 'DEG_kWh_lag_29'] = sg(df[index].DEG_kWh, 29, p)
    return df


def lgbm_regression_efecto_acumulado_con_linea_base_primer_experimento(data_mean,
                                                                       numerical_features_list,
                                                                       categorical_features_list,
                                                                       interventions,
                                                                       stacking,
                                                                       get_best_parameters=False):
    params1 = {'learning_rate': 0.001,
               'num_leaves': 4,
               'max_depth': -1,
               'min_data_in_leaf': 10}
    params2 = {'learning_rate': 0.001,
               'num_leaves': 4,
               'max_depth': -1,
               'min_data_in_leaf': 10}
    param3 = {'learning_rate': 0.001,
              'num_leaves': 4,
              'max_depth': -1,
              'min_data_in_leaf': 10}
    counter = 0
    for experiment, control, params_fixed in zip([1, 2, 3], ['1CONTROL', '2CONTROL', '3CONTROL'],
                                                 [params1, params2, param3]):
        intervention_data = interventions[experiment]
        pre_period = intervention_data[1]
        post_period = intervention_data[2]
        range_post_intervention_period = pd.date_range(start=post_period[0], end=post_period[1], freq='D')

        if counter == 0:
            range_pre_intervention_period = pd.date_range(start=pre_period[0], end=pre_period[1], freq='D')

        # trick because onece we eliminate values the groups keep storing the info (stupid python)
        columns = data_mean.columns
        array = data_mean[(data_mean.timestamp.isin(
            range_pre_intervention_period))].values  # | (data_mean.INTERVENTION == control)].values
        data_train = pd.DataFrame(array, index=array[:, 0], columns=columns)
        data_train = data_train.reset_index(drop=True)

        if get_best_parameters:
            array = data_mean[(data_mean.INTERVENTION == control)].values
            data_test = pd.DataFrame(array, index=array[:, 0], columns=columns)
            data_test = data_test.reset_index(drop=True)
        else:
            data_test = pd.DataFrame()

        models, features, categorical_features_list = model_train(data_train, numerical_features_list,
                                                                  categorical_features_list, params_fixed, stacking,
                                                                  data_test=data_test)
        final_model = models[0]

        for c in categorical_features_list:
            data_mean[c] = data_mean[c].astype('category')

        data_pre = data_mean[
            (data_mean.timestamp.isin(range_pre_intervention_period)) & (data_mean.EXPERIMENT == experiment)]
        data_mean.loc[data_mean['timestamp'].isin(range_pre_intervention_period) & (
                data_mean.EXPERIMENT == experiment), 'GBM_consumption_kWh'] = np.exp(
            final_model.predict(data_pre[features])) - 1

        data_post = data_mean[
            (data_mean.timestamp.isin(range_post_intervention_period)) & (data_mean.EXPERIMENT == experiment)]
        data_mean.loc[data_mean['timestamp'].isin(range_post_intervention_period) & (
                data_mean.EXPERIMENT == experiment), 'GBM_consumption_kWh'] = np.exp(
            final_model.predict(data_post[features])) - 1
        counter += 1
    return data_mean


def lgbm_regression_efecto_acumulado_con_linea_base(data_mean,
                                                    params,
                                                    numerical_features_list,
                                                    categorical_features_list,
                                                    interventions,
                                                    stacking,
                                                    get_best_parameters=False):
    counter = 0
    for experiment, control, params_fixed in zip([1, 2, 3], ['1CONTROL', '2CONTROL', '3CONTROL'],
                                                 params):
        intervention_data = interventions[experiment]
        pre_period = intervention_data[1]
        post_period = intervention_data[2]
        range_pre_intervention_period = pd.date_range(start=pre_period[0], end=pre_period[1], freq='D')
        range_post_intervention_period = pd.date_range(start=post_period[0], end=post_period[1], freq='D')

        # trick because onece we eliminate values the groups keep storing the info (stupid python)
        if counter == 0:
            columns = data_mean.columns
            array = data_mean[(data_mean.timestamp.isin(
                range_pre_intervention_period))].values  # | (data_mean.INTERVENTION == control)].values
            data_train = pd.DataFrame(array, index=array[:, 0], columns=columns)
            data_train = data_train.reset_index(drop=True)

            if get_best_parameters:
                array = data_mean[(data_mean.INTERVENTION == control)].values
                data_test = pd.DataFrame(array, index=array[:, 0], columns=columns)
                data_test = data_test.reset_index(drop=True)
            else:
                data_test = pd.DataFrame()

            models, features, categorical_features_list = model_train(data_train, numerical_features_list,
                                                                      categorical_features_list, params_fixed, stacking,
                                                                      data_test=data_test)
            # save for use in next iterations
            range_pre_intervention_period_no_one = range_pre_intervention_period

        elif counter == 1:
            columns = data_mean.columns
            array = data_mean[(data_mean.timestamp.isin(range_pre_intervention_period))].values
            data_train = pd.DataFrame(array, index=array[:, 0], columns=columns)
            data_train = data_train.reset_index(drop=True)

            if get_best_parameters:
                array = data_mean[(data_mean.INTERVENTION == control)].values
                data_test = pd.DataFrame(array, index=array[:, 0], columns=columns)
                data_test = data_test.reset_index(drop=True)
            else:
                data_test = pd.DataFrame()

            models, features, categorical_features_list = model_train(data_train, numerical_features_list,
                                                                      categorical_features_list, params_fixed, stacking,
                                                                      data_test=data_test)

        else:
            # include the values of first experiment for training
            columns = data_mean.columns
            array = data_mean[(data_mean.timestamp.isin(range_pre_intervention_period)) | (data_mean.timestamp.isin(
                range_pre_intervention_period_no_one))].values  # | (data_mean.INTERVENTION == control)].values
            data_train = pd.DataFrame(array, index=array[:, 0], columns=columns)
            data_train = data_train.reset_index(drop=True)

            if get_best_parameters:
                array = data_mean[(data_mean.INTERVENTION == control)].values
                data_test = pd.DataFrame(array, index=array[:, 0], columns=columns)
                data_test = data_test.reset_index(drop=True)
            else:
                data_test = pd.DataFrame()

            models, features, categorical_features_list = model_train(data_train, numerical_features_list,
                                                                      categorical_features_list, params_fixed, stacking,
                                                                      data_test=data_test)

        if get_best_parameters == False:
            final_model = models[0]

            for c in categorical_features_list:
                data_mean[c] = data_mean[c].astype('category')

            data_pre = data_mean[
                (data_mean.timestamp.isin(range_pre_intervention_period)) & (data_mean.EXPERIMENT == experiment)]
            data_mean.loc[data_mean['timestamp'].isin(range_pre_intervention_period) & (
                    data_mean.EXPERIMENT == experiment), 'GBM_consumption_kWh'] = np.exp(
                final_model.predict(data_pre[features])) - 1

            data_post = data_mean[
                (data_mean.timestamp.isin(range_post_intervention_period)) & (data_mean.EXPERIMENT == experiment)]
            data_mean.loc[data_mean['timestamp'].isin(range_post_intervention_period) & (
                    data_mean.EXPERIMENT == experiment), 'GBM_consumption_kWh'] = np.exp(
                final_model.predict(data_post[features])) - 1

        counter += 1
    return data_mean
