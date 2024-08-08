import sys
import pandas as pd
import optuna
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import holidays
from datetime import date

def readCSV(file):
    if file =="clean_consumos.csv":
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]

        df['data_saida'] = pd.to_datetime(df['data_saida'])

        mask_lisboa = (df['cst'] == 'Lisboa') & (df['instituicao'].isin([1060, 1061]))
        mask_porto = (df['cst'] == 'Porto') & (df['instituicao'].isin([1061, 1062]))
        mask_coimbra = (df['cst'] == 'Coimbra') & (df['instituicao'].isin([1060, 1062]))

        combined_mask = (mask_lisboa | mask_porto | mask_coimbra ) & (df['data_saida'] < '2017-01-01')

        df = df[~combined_mask]

        return (df, "consumos")
    else:
        df = pd.read_csv(file, sep=';')

        mask_approved_collection = (df['colheita_fase'] == 'E') & (df['colheita_estado'] == 'O')

        df = df[mask_approved_collection]
    
        return (df, "colheitas")
    
def filterData(df_tuple, years):
    df, df_type = df_tuple  # Unpack the tuple into separate variables

    reference_date = pd.Timestamp('2020-01-01')

    # Subtract X years from the reference date
    X_years_before_2020 = reference_date - pd.DateOffset(years=int(years))
    
    # Rename column colheita_data of dataset Colheitas
    if df_type == "colheitas":
        df = df.rename(columns={'colheita_data': 'data_saida'})
        df['data_saida'] = pd.to_datetime(df['data_saida'])

    combined_mask = (df['data_saida'] >= X_years_before_2020) & (df['data_saida'] < reference_date)

    df_last_X_years = df[combined_mask]

    return df_last_X_years

def dataset_preparation(df):

    def check_holidays(date):
        # Specify the country code for Portugal
        country_code = 'PT'

        # Create a holiday object for Portugal
        pt_holidays = holidays.CountryHoliday(country_code)

        # Check if the date is a holiday in Portugal
        if date in pt_holidays:
            return 1
        else:
            return 0


    def get_season(month):
        if month in [12, 1, 2]:
            return 1 # Winter
        elif month in [3, 4, 5]:
            return 2 # Spring
        elif month in [6, 7, 8]:
            return 3 # Summer
        elif month in [9, 10, 11]:
            return 4 # Autumn

    df['year'] = df['data_saida'].dt.year
    df['month'] = df['data_saida'].dt.month
    df['day'] = df['data_saida'].dt.day
    df['day_of_week'] = df['data_saida'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) # Working Day = 0, Weekend = 1
    df['day_of_year'] = df['data_saida'].dt.dayofyear
    df['week_number'] = df['data_saida'].dt.isocalendar().week
    df['season'] = df['month'].apply(get_season)
    df['holiday'] = df['data_saida'].apply(check_holidays)

    return df

def split_sequence(dataset_ready):
    dataset_train = dataset_ready[dataset_ready['year'] < 2019]
    dataset_test = dataset_ready[dataset_ready['year'] >= 2019]

    X_train = dataset_train.loc[:, ~dataset_train.columns.isin(['count', 'data_saida'])]
    X_test = dataset_test.loc[:, ~dataset_test.columns.isin(['count', 'data_saida'])]
    y_train = dataset_train['count']
    y_test = dataset_test['count']
    test_dates = dataset_test['data_saida']

    return X_train, X_test, y_train, y_test, test_dates


def resample_by_period_of_Days(pred_df, test_df, days):
    
    #Make groups of X days, if the last group doesn't match the group size( X days) it discards because it is an incomplete group

    i = 1
    mapes = []
    tmp_pred = pd.Series()
    tmp_test = pd.Series()
    for index in range(len(pred_df)):
        if i > days:
            mapes.append(mean_absolute_percentage_error(tmp_test, tmp_pred))

            #reset group
            tmp_pred = pd.Series()
            tmp_test = pd.Series()
            i = 1
        else:
            tmp_pred = pd.concat([tmp_pred, pd.Series(pred_df.iloc[index]['predicted_count'])])
            tmp_test = pd.concat([tmp_test, pd.Series(test_df.iloc[index]['test_count'])])
            i += 1
    return mapes


def predict_function(trial, X_train, X_test, y_train, y_test, test_dates, filename):
    possibilities = ['relu', 'sigmoid', 'tanh', 'softmax']
    activation_layer = trial.suggest_categorical('activation_layer', possibilities)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'adagrad'])

    # Convert DataFrames to NumPy arrays
    X_train_array = X_train.to_numpy().astype('float32')
    X_test_array = X_test.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    y_test = y_test.to_numpy().astype('float32')

    # Reshape input data for LSTM (number of samples, number of timesteps, number of features)
    n_steps = 1  # Assuming each sample is one day
    n_features = X_train.shape[1]  # Number of features
    X_train_lstm = X_train_array.reshape((X_train_array.shape[0], n_steps, n_features))
    X_test_lstm = X_test_array.reshape((X_test_array.shape[0], n_steps, n_features))

    model = Sequential()
    model.add(LSTM(units=128, activation=activation_layer))
    model.add(Dense(1))
    model.compile(optimizer=optimizer_name, loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_lstm, y_train, epochs=370, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test_lstm)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])
    test_df["test_count"].replace(0, 1, inplace=True)

    if filename == "clean_consumos.csv":
        mapes = resample_by_period_of_Days(pred_df, test_df, 14) #2 weeks
    else:
        mapes = resample_by_period_of_Days(pred_df, test_df, 42) #6 weeks

    return np.mean(mapes)


def predict_hidden_units(trial, X_train, X_test, y_train, y_test, test_dates, filename):
    n_units_layer = trial.suggest_int('n_units_layer', 64, 256, step=8)
    n_epochs = trial.suggest_int('n_epochs',300, 500, step=10)


    # Convert DataFrames to NumPy arrays
    X_train_array = X_train.to_numpy().astype('float32')
    X_test_array = X_test.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    y_test = y_test.to_numpy().astype('float32')

    # Reshape input data for LSTM (number of samples, number of timesteps, number of features)
    n_steps = 1  # Assuming each sample is one day
    n_features = X_train.shape[1]  # Number of features
    X_train_lstm = X_train_array.reshape((X_train_array.shape[0], n_steps, n_features))
    X_test_lstm = X_test_array.reshape((X_test_array.shape[0], n_steps, n_features))

    model = Sequential()
    model.add(LSTM(units=n_units_layer, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_lstm, y_train, epochs=n_epochs, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test_lstm)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])
    test_df["test_count"].replace(0, 1, inplace=True)

    if filename == "clean_consumos.csv":
        mapes = resample_by_period_of_Days(pred_df, test_df, 14) #2 weeks
    else:
        mapes = resample_by_period_of_Days(pred_df, test_df, 42) #6 weeks

    return np.mean(mapes)


def drawGraph(mapes):

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


    plt.plot(months, mapes, marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python MLP_evaluation.py <file>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: clean_consumos.csv or colheitas_10-05-2021.csv")
        sys.exit(1)

    # Parse command-line arguments
    file = sys.argv[1]
    years = 10
    df = readCSV(file)

    print("Li a data")

    df_X_years = filterData(df, years)
    print("Filtrei a data")

    group_by_Day = df_X_years.groupby(pd.Grouper(key='data_saida', freq="D")).size().reset_index(name='count')
    print("Agrupei")

    dataset_ready = dataset_preparation(group_by_Day)

    X_train, X_test, y_train, y_test, test_dates = split_sequence(dataset_ready) 

    #mapes_sum = predict(X_train, X_test, y_train, y_test, test_dates, file)

    if file =="clean_consumos.csv":
        storage_url = "sqlite:///LSTM_consumption.sqlite3"
    else:
        storage_url = "sqlite:///LSTM_collection.sqlite3"
    
    study_name = "Hidden Layers and Units Evaluation"
    #study_name = "Activation Function and Optimizer Evaluation"

    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print("Study deleted successfully.")
    except KeyError:
        print("Study does not exist.")

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize')
    study.optimize(lambda trial: predict_hidden_units(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=500)
    #study.optimize(lambda trial: predict_function(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=100)
    #study.optimize(lambda trial: predict_optimizer(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=20)
    #study.optimize(lambda trial: predict_epochs(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))