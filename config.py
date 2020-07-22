import os

DIRECTORY = os.path.abspath(os.path.dirname(__file__))

##FOLDERS
DATA_RAW_FOLDER = os.path.join(DIRECTORY, "data", "data_raw")
DATA_CONSUMPTION_RAW_FOLDER = os.path.join(DATA_RAW_FOLDER, "CONSUMPTION")
DATA_WEATHER_RAW_FOLDER = os.path.join(DATA_RAW_FOLDER, "WEATHER")
DATA_BIM_RAW_FOLDER = os.path.join(DATA_RAW_FOLDER, "BIM")

##FILES RAW
DATA_WEATHER_RAW_MAIN_FILE = os.path.join(DATA_WEATHER_RAW_FOLDER, "weatherdata_S44.csv")
DATA_WEATHER_RAW_AUXILIARY_FILE = os.path.join(DATA_WEATHER_RAW_FOLDER, "weatherdata_S24.csv")

# FOLDERS PROCESSED
DATA_PROCESSED_FOLDER = os.path.join(DIRECTORY, "data", "data_processed")
DATA_HOLIDAYS_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "HOLIDAYS")
DATA_RESULTS_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "RESULTS")
DATA_CONSUMPTION_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "CONSUMPTION")
DATA_WEATHER_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "WEATHER")
DATA_METADATA_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "METADATA")
DATA_SOLAR_GAINS_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "SOLAR_GAINS")
DATA_ENTHALPY_GRADIENTS_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "ENTHALPY_GRADIENTS")
DATA_ISO_CONSUMPTION_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "ISO_CONSUMPTION")
DATA_GBM_CONSUMPTION_PROCESSED_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "GBM_CONSUMPTION")
DATA_VACATIONS_FOLDER = os.path.join(DATA_PROCESSED_FOLDER, "VACATIONS")

##FILES PROCESSED
DATA_BSTS_STATS_FILE = os.path.join(DATA_RESULTS_FOLDER, "BSTS_STATS.csv")
DATA_CONSUMPTION_SEMI_PROCESSED_FILE = os.path.join(DATA_CONSUMPTION_PROCESSED_FOLDER, "consumption_semi_processed.csv")
DATA_VACATIONS_FILE = os.path.join(DATA_VACATIONS_FOLDER, "vacations.csv")
DATA_VACATIONS_INTERVENTION_FILE = os.path.join(DATA_VACATIONS_FOLDER, "vacations_per_intervention.csv")
DATA_MONTHLY_MEAN_NORMALIZED_FILE = os.path.join(DATA_BIM_RAW_FOLDER, "monthly_mean_control.csv")
DATA_CONSUMPTION_PROCESSED_FILE = os.path.join(DATA_CONSUMPTION_PROCESSED_FOLDER, "consumption.csv")
DATA_METADATA_PROCESSED_FILE = os.path.join(DATA_METADATA_PROCESSED_FOLDER, "METADATA.xlsx")
DATA_HOLIDAYS_PROCESSED_FILE = os.path.join(DATA_HOLIDAYS_PROCESSED_FOLDER, "holidays.csv")
DATA_WEATHER_PROCESSED_FILE = os.path.join(DATA_WEATHER_PROCESSED_FOLDER, "weather.csv")
DATA_ENTHALPY_GRADIENTS_PROCESSED_FILE = os.path.join(DATA_ENTHALPY_GRADIENTS_PROCESSED_FOLDER,
                                                      "enthalpy_gradients.csv")
DATA_SOLAR_GAINS_PROCESSED_FILE = os.path.join(DATA_SOLAR_GAINS_PROCESSED_FOLDER, "solar_gains.csv")
DATA_ISO_CONSUMPTION_PROCESSED_FILE = os.path.join(DATA_ISO_CONSUMPTION_PROCESSED_FOLDER, "consumption.csv")
DATA_GBM_CONSUMPTION_PROCESSED_FILE = os.path.join(DATA_GBM_CONSUMPTION_PROCESSED_FOLDER, "consumption.csv")
DATA_GBM_CONSUMPTION_PROCESSED_FILE_AGGREGATED = os.path.join(DATA_GBM_CONSUMPTION_PROCESSED_FOLDER,
                                                              "consumption_agg.csv")
BEST_PARAMETERS_FILE = os.path.join(DIRECTORY, "best_parameters.json")
BEST_FEATURES_FILE = os.path.join(DIRECTORY, "best_features.json")
# Intervention calendar
INTERVENTION_CALENDAR = {1: [['1T1', '1T2', '1T3'],
                             ['2018-02-24', '2018-04-24'],
                             ['2018-04-25', '2018-10-25'],
                             ['2018-07-25']],
                         2: [['2T4', '2T5'],
                             ['2018-08-24', '2018-10-24'],
                             ['2018-10-25', '2019-03-25'],
                             ['2018-12-24']],
                         3: [['3T1', '3T2', '3T3'],
                             ['2019-01-24', '2019-03-24'],
                             ['2019-03-25', '2019-09-25'],
                             ['2019-06-25']]}

SEED = 12345

DYNAMIC_SPACE = {"learning_rate": [0.1, 0.2, 0.3],
                 "num_leaves": [16, 31, 64],
                 "num_iterations": [1000, 5000, 10000],
                 "feature_fraction": [0.2, 0.4, 0.8],
                 "bagging_fraction": [0.2, 0.4, 0.8],
                 "min_data_in_leaf": [10, 20, 64],
                 "objective": ["quantile"],
                 "metric": ["quantile"]}

EXPERIMENTS = [1, 2, 3]
CONTROL_GROUPS = ['1CONTROL', '2CONTROL', '3CONTROL']

ALL_NUMERICAL_FEATURES = [
    "ISO_kWh",
    "DEG_C_kWh",
    "SG_kWh",
    "ISO_kWh_smooth_29",
    "SG_kWh_smooth_29",
    "DEG_C_kWh_smooth_29",
    "ISO_kWh_smooth_89",
    "SG_kWh_smooth_89",
    "DEG_C_kWh_smooth_89",
    'EMPTY_DAY_MEAN',
    'FULL_DAY_MEAN',

]

ALL_CATEGORICAL_FEATURES = [
    "EMPTY_DAY_INT_0.1",
    "FULL_DAY_INT_0.9",
    "EMPTY_DAY_INT_0.25",
    "FULL_DAY_INT_0.75",
    'GFA_m2_MEAN',
    'BEDROOMS_MEAN',
    "INCOME_MEAN",
    'teaching_time',
    'weekday',
    'month',
    'year',
    'dayofweek',
    'INTERVENTION'
]
