import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.layers import LSTM
import matplotlib.pyplot as plt
import holidays
from datetime import date
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


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
    X_years_before_2020 = reference_date - pd.DateOffset(years=years)
    
    # Rename column colheita_data of dataset Colheitas
    if df_type == "colheitas":
        df = df.rename(columns={'colheita_data': 'data_saida'})
        df['data_saida'] = pd.to_datetime(df['data_saida'])

    combined_mask = (df['data_saida'] >= X_years_before_2020) & (df['data_saida'] < reference_date)

    df_last_X_years = df[combined_mask]

    df_last_X_years.to_csv('filtered_consumos_data.csv', index=False) 

    return df_last_X_years

def filterExactDate(group_by_Day, day_of_week):
    
    #O teste começa de dia 4 de março (segunda feira) a 28 de abril (domingo) (8 semanas de range)
    test_end_date = pd.Timestamp('2019-04-29') + pd.Timedelta(days=day_of_week)
    test_begin_date = pd.Timestamp('2019-03-03') + pd.Timedelta(days=day_of_week)

    combined_mask_train = (group_by_Day['data_saida'] <= test_begin_date)
    combined_mask_test = (group_by_Day['data_saida'] > test_begin_date) & (group_by_Day['data_saida'] < test_end_date)
    test_data = group_by_Day[combined_mask_test]
    train_data = group_by_Day[combined_mask_train]

    return train_data["count"], test_data["count"]

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


def split_sequence_MLP(dataset_ready):
    dataset_train = dataset_ready[dataset_ready['year'] < 2019]
    dataset_test = dataset_ready[(dataset_ready['data_saida'] > pd.Timestamp('2019-03-03')) & (dataset_ready['data_saida'] < pd.Timestamp('2019-03-18'))]

    X_train = dataset_train.loc[:, ~dataset_train.columns.isin(['count', 'data_saida'])]
    X_test = dataset_test.loc[:, ~dataset_test.columns.isin(['count', 'data_saida'])]
    y_train = dataset_train['count']
    y_test = dataset_test['count']
    test_dates = dataset_test['data_saida']

    X_train = X_train.to_numpy().astype('float32')
    X_test = X_test.to_numpy().astype('float32')
    y_train = y_train.to_numpy().astype('float32')
    y_test = y_test.to_numpy().astype('float32')

    return X_train, X_test, y_train, y_test, test_dates


def split_sequence_LSTM(dataset_ready):
    dataset_train = dataset_ready[dataset_ready['year'] < 2019]
    dataset_test = dataset_ready[(dataset_ready['data_saida'] > pd.Timestamp('2019-03-03')) & (dataset_ready['data_saida'] < pd.Timestamp('2019-03-18'))]

    X_train = dataset_train.loc[:, ~dataset_train.columns.isin(['count', 'data_saida'])]
    X_test = dataset_test.loc[:, ~dataset_test.columns.isin(['count', 'data_saida'])]
    y_train = dataset_train['count']
    y_test = dataset_test['count']
    test_dates = dataset_test['data_saida']

    return X_train, X_test, y_train, y_test, test_dates

def forecast_HW(train_data, increment, seasonal_periods):
    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=increment)
    return prediction


def predict(group_by_Day, file):

    if file == "clean_consumos.csv":
        days = [14] # 2 weeks
    else:
        days = [42] # 6 weeks

    # Create a list of subcolumn names
    sub_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_dict = {day: i for i, day in enumerate(sub_columns)}

    predictions = pd.DataFrame(columns=sub_columns)
    mapes = pd.DataFrame(columns=sub_columns)
    seasonal_periods = 7

    for days_of_week in week_dict:
        train_data, test_data = filterExactDate(group_by_Day, week_dict[days_of_week])
        print("FilterExactDate " + str(days_of_week))

        pred_temp = []
        mape_temp = []

        for day in days:
    
            test_data_local = test_data.iloc[:day]
                
            prediction = forecast_HW(train_data, day, seasonal_periods)
               
            pred_temp.append(prediction.values[0])
            mape_temp.append(mape(test_data_local,prediction))
        
        predictions[days_of_week] = pred_temp
        mapes[days_of_week] = mape_temp

    return predictions, mapes

def predict_MLP(X_train, X_test, y_train, y_test, test_dates, file):
    if file == "clean_consumos.csv":
        # Build the model
        model = Sequential()
        # Hidden Layer 0
        model.add(Dense(120, activation='relu', input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(16, activation='relu'))
        # Hidden Layer 2
        model.add(Dense(128, activation='relu'))
        # Hidden Layer 3
        model.add(Dense(96, activation='relu'))
        # Hidden Layer 4
        model.add(Dense(120, activation='relu'))
        # Hidden Layer 5
        model.add(Dense(128, activation='relu'))
        # Output Layer
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=80, verbose=0)
    
    else:
        # Build the model
        model = Sequential()
        # Hidden Layer 0
        model.add(Dense(128, activation='softmax', input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(120, activation='softmax'))
        # Hidden Layer 2
        model.add(Dense(88, activation='softmax'))
        # Hidden Layer 3
        model.add(Dense(96, activation='softmax'))
        # Hidden Layer 4
        model.add(Dense(128, activation='softmax'))
        # Output Layer
        model.add(Dense(1))

        model.compile(optimizer='rmsprop', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=5, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])
    test_df["test_count"].replace(0, 1, inplace=True)
    
    return  mape(test_df['test_count'], pred_df['predicted_count'])


def predict_LSTM(X_train, X_test, y_train, y_test, test_dates, file):
    
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

    if file == "clean_consumos.csv":
        model = Sequential()
        model.add(LSTM(units=128, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train_lstm, y_train, epochs=370, verbose=0)
    else:
        model = Sequential()
        model.add(LSTM(units=232, activation='softmax'))
        model.add(Dense(1))
        model.compile(optimizer='adagrad', loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train_lstm, y_train, epochs=480, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test_lstm)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])
    test_df["test_count"].replace(0, 1, inplace=True)

    return  mape(test_df['test_count'], pred_df['predicted_count'])


def mape(test_data, prediction):
        
    subtract = (test_data - prediction)
    mape = np.mean(np.abs((subtract) / test_data)) * 100
    return mape

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python Traditional_vs_Non-Traditional.py <file>")
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

    predictions, mapes = predict(group_by_Day, file)

    mape_HW = mapes.sum(axis=1).iloc[0] / len(mapes.columns)
    print("Calculei MAPE HW")

    #MLP and LSTM preparation
    dataset_ready = dataset_preparation(group_by_Day)

    X_train_MLP, X_test_MLP, y_train_MLP, y_test_MLP, test_dates_MLP = split_sequence_MLP(dataset_ready) 

    X_train_LSTM, X_test_LSTM, y_train_LSTM, y_test_LSTM, test_dates_LSTM = split_sequence_LSTM(dataset_ready) 


    mape_MLP = predict_MLP(X_train_MLP, X_test_MLP, y_train_MLP, y_test_MLP, test_dates_MLP, file)
    print("Calculei MAPE MLP")
    
    mape_LSTM = predict_LSTM(X_train_LSTM, X_test_LSTM, y_train_LSTM, y_test_LSTM, test_dates_LSTM, file)
    print("Calculei MAPE LSTM")


