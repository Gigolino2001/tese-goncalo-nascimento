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


def dataset_preparation(df):

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
    df['season'] = df['month'].apply(get_season)

    return df

def split_sequence(dataset_ready, columns_exclude):
    dataset_train = dataset_ready[dataset_ready['year'] < 2018]
    dataset_test = dataset_ready[dataset_ready['year'] >= 2018]

    X_train = dataset_train.loc[:, ~dataset_train.columns.isin(['count', 'data_saida'])]
    X_test = dataset_test.loc[:, ~dataset_test.columns.isin(['count', 'data_saida'])]
    X_train = X_train.drop(columns=columns_exclude)
    X_test = X_test.drop(columns=columns_exclude)
    y_train = dataset_train['count']
    y_test = dataset_test['count']
    test_dates = dataset_test['data_saida']

    return X_train, X_test, y_train, y_test, test_dates


def predict(X_train, X_test, y_train, y_test, test_dates, filename):

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


    if filename == "concelho_metropolitana_lisboa.csv":
        model = Sequential()
        model.add(LSTM(units=32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train_lstm, y_train, epochs=140, verbose=0)
    
    else:
        model = Sequential()
        model.add(LSTM(units=8, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train_lstm, y_train, epochs=10, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test_lstm)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])

    return mean_absolute_percentage_error(test_df['test_count'], pred_df['predicted_count'])


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python MLP_evaluation.py <file>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: concelho_metropolitana_lisboa.csv or concelho_metropolitana_porto.csv")
        print("- <file2>: The name of the CSV file. Valid values: weather_lisbon.csv or weather_porto.csv")
        sys.exit(1)

    # Parse command-line arguments
    file = sys.argv[1]
    file2 = sys.argv[2]

    df = pd.read_csv(file)
    df2 = pd.read_csv(file2)
    df2.rename(columns={'date_time': 'data_saida'}, inplace=True)
    print("Li a data")

    df['data_saida'] = pd.to_datetime(df['data_saida'])
    df2['data_saida'] = pd.to_datetime(df2['data_saida'])
    group_by_Month = df.groupby(pd.Grouper(key='data_saida', freq="M")).size().reset_index(name='count')
    print("Agrupei")




    dataset_ready = dataset_preparation(group_by_Month)
    pivot = pd.pivot_table(df2, 
                       index=pd.Grouper(key='data_saida', freq="M"),
                       values=['mintempC', 'maxtempC', 'precipMM', 'totalSnow_cm', 'sunHour', 'uvIndex', 'moon_illumination', 'DewPointC', 'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph', 'cloudcover', 'humidity', 'pressure', 'visibility', 'winddirDegree', 'windspeedKmph'], 
                       aggfunc={'mintempC': 'mean', 'maxtempC': 'mean', 'precipMM': 'sum', 'totalSnow_cm': 'sum', 'sunHour': 'mean', 'uvIndex': 'mean', 'moon_illumination': 'mean', 'DewPointC': 'mean', 'FeelsLikeC': 'mean', 'HeatIndexC': 'mean', 'WindChillC': 'mean', 'WindGustKmph': 'mean', 'cloudcover': 'mean', 'humidity': 'mean', 'pressure': 'mean', 'visibility': 'mean', 'winddirDegree': 'mean', 'windspeedKmph': 'mean'})


    merged_df = pd.merge(dataset_ready, pivot, on='data_saida', how='left')

    print("Merged")
    temporal_features = ['year', 'month', 'day', 'season', 'data_saida', 'count']
    
    columns = merged_df.columns.to_list()
    results = {}
    #weather_columns = list(filter(lambda x: x not in temporal_features, columns))
    weather_columns = ['all']
    for col in weather_columns:
        #columns_exclude = list(filter(lambda x: x != col, weather_columns))
        columns_exclude = []
        X_train, X_test, y_train, y_test, test_dates = split_sequence(dataset_ready, columns_exclude) 
    
        mapes = []
        for x in range(0,100):
            mapes.append(predict(X_train, X_test, y_train, y_test, test_dates, file))

        end = np.mean(mapes)
        print(str(col) + " : " + str(end))
        results[col] = end
    print("ACABEI")
    print(results)