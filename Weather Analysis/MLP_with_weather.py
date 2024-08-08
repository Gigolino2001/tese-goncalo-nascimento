import sys
import pandas as pd
import optuna
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import holidays
from datetime import date

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

    X_train = X_train.to_numpy().astype('float32')
    X_test = X_test.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    y_test = y_test.to_numpy().astype('float32')

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

def predict(X_train, X_test, y_train, y_test, test_dates, file):

    # Build the model
    model = Sequential()

    if file == "concelho_metropolitana_lisboa.csv":
        # Hidden Layer 0
        model.add(Dense(8, activation="softmax", input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(48, activation="softmax"))

    else:
        model.add(Dense(104, activation="softmax", input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(64, activation="softmax"))
        # Hidden Layer 2
        model.add(Dense(8, activation="softmax"))

        print("PORTO")

    # Output Layer
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')


    history = model.fit(X_train, y_train, epochs=4, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])
    test_df["test_count"].replace(0, 1, inplace=True)


    mapes = resample_by_period_of_Days(pred_df, test_df, 42) #6 weeks

    return np.mean(mapes)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python MLP_without_weather.py <file>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: concelho_metropolitana_lisboa.csv or concelho_metropolitana_porto.csv")
        print("- <file2>: The name of the CSV file. Valid values: weather_lisbon.csv or weather_porto.csv")
        sys.exit(1)
    
    file = sys.argv[1]
    file2 = sys.argv[2]

    df = pd.read_csv(file)
    df2 = pd.read_csv(file2)
    df2.rename(columns={'date_time': 'data_saida'}, inplace=True)
    print("Li a data")

    df['data_saida'] = pd.to_datetime(df['data_saida'])
    df2['data_saida'] = pd.to_datetime(df2['data_saida'])
    group_by_Day = df.groupby(pd.Grouper(key='data_saida', freq="D")).size().reset_index(name='count')
    print("Agrupei")

    merged_df = pd.merge(group_by_Day, df2, on='data_saida', how='left')
    print("Merged")

    dataset_ready = dataset_preparation(merged_df)

    dataset_ready = dataset_ready.drop(columns=
                                       ['moonrise', 'moonset', 'sunrise', 'sunset', 'tempC', 'uvIndex.1'])

    temporal_features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'day_of_year', 'week_number', 'season',
                         'holiday', 'data_saida', 'count']
    
    columns = dataset_ready.columns.to_list()

    weather_columns = list(filter(lambda x: x not in temporal_features, columns))
    for col in weather_columns:
        columns_exclude = list(filter(lambda x: x != col, weather_columns))
        X_train, X_test, y_train, y_test, test_dates = split_sequence(dataset_ready, columns_exclude) 
    
        mapes = []
        for x in range(0,100):
            mapes.append(predict(X_train, X_test, y_train, y_test, test_dates, file))

        end = np.mean(mapes)
        print(str(col) + " : " + str(end))
    print("ACABEI")