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
    df['week_number'] = df['data_saida'].dt.isocalendar().week
    df['season'] = df['month'].apply(get_season)

    return df

def split_sequence(dataset_ready):
    dataset_train = dataset_ready[dataset_ready['year'] < 2018]
    dataset_test = dataset_ready[dataset_ready['year'] >= 2018]

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


def predict_epochs(trial, X_train, X_test, y_train, y_test, test_dates, file):

    num_epochs = trial.suggest_int('num_epochs', 1, 100)

    # Build the model
    model = Sequential()

    if file == "concelho_metropolitana_lisboa.csv":
        # Hidden Layer 0
        model.add(Dense(104, activation='relu', input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(112, activation='relu'))
        # Hidden Layer 2
        model.add(Dense(32, activation='relu'))
        # Hidden Layer 3
        model.add(Dense(112, activation='relu'))
        # Hidden Layer 4
        model.add(Dense(48, activation='relu'))
        # Hidden Layer 5
        model.add(Dense(120, activation='relu'))

        print("LISBOA")
    else:
        model.add(Dense(104, activation="softmax", input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(64, activation="softmax"))
        # Hidden Layer 2
        model.add(Dense(8, activation="softmax"))

        print("PORTO")
    # Output Layer
    model.add(Dense(1))


    model.compile(optimizer="rmsprop", loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=num_epochs, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])


    return mean_absolute_percentage_error(test_df['test_count'], pred_df['predicted_count'])


def predict_optimizer(trial, X_train, X_test, y_train, y_test, test_dates, file):

    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'adagrad'])

    # Build the model
    model = Sequential()

    if file == "concelho_metropolitana_lisboa.csv":

        model.add(Dense(104, activation='relu', input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(112, activation='relu'))
        # Hidden Layer 2
        model.add(Dense(32, activation='relu'))
        # Hidden Layer 3
        model.add(Dense(112, activation='relu'))
        # Hidden Layer 4
        model.add(Dense(48, activation='relu'))
        # Hidden Layer 5
        model.add(Dense(120, activation='relu'))

        print("LISBOA")
    else:
        model.add(Dense(104, activation="softmax", input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(64, activation="softmax"))
        # Hidden Layer 2
        model.add(Dense(8, activation="softmax"))

        print("PORTO")
    # Output Layer
    model.add(Dense(1))


    model.compile(optimizer=optimizer_name, loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Predict on the test set
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])

    return mean_absolute_percentage_error(test_df['test_count'], pred_df['predicted_count'])


def predict_function(trial, X_train, X_test, y_train, y_test, test_dates, file):
    possibilities = ['relu', 'sigmoid', 'tanh', 'softmax']
    activation_layer = trial.suggest_categorical('activation_layer', possibilities)

    # Build the model
    model = Sequential()

    if file == "concelho_metropolitana_lisboa.csv":
        # Hidden Layer 0
        model.add(Dense(104, activation=activation_layer, input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(112, activation=activation_layer))
        # Hidden Layer 2
        model.add(Dense(32, activation=activation_layer))
        # Hidden Layer 3
        model.add(Dense(112, activation=activation_layer))
        # Hidden Layer 4
        model.add(Dense(48, activation=activation_layer))
        # Hidden Layer 5
        model.add(Dense(120, activation=activation_layer))
        print("LISBOA")
    else:
        model.add(Dense(104, activation=activation_layer, input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(64, activation=activation_layer))
        # Hidden Layer 2
        model.add(Dense(8, activation=activation_layer))

        print("PORTO")

    # Output Layer
    model.add(Dense(1))

    model.compile(optimizer="adam", loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])


    return mean_absolute_percentage_error(test_df['test_count'], pred_df['predicted_count'])



def predict_hidden_units(trial, X_train, X_test, y_train, y_test, test_dates):
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 6)
    hidden_units = [trial.suggest_int(f'n_units_layer_{i}', 1, 16) * 8 for i in range(num_hidden_layers)]

    # Build the model
    model = Sequential()
    model.add(Dense(hidden_units[0], activation='relu', input_dim=X_train.shape[1]))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, verbose=0)

    # Predict on the test set
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame(y_pred, index=test_dates, columns=['predicted_count'])
    test_df = pd.DataFrame(y_test, index=test_dates, columns=['test_count'])
    

    return mean_absolute_percentage_error(test_df['test_count'], pred_df['predicted_count'])


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python MLP_without_weather_monthly.py <file>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: concelho_metropolitana_lisboa.csv or concelho_metropolitana_porto.csv")
        sys.exit(1)
    
    file = sys.argv[1]

    df = pd.read_csv(file)
    print("Li a data")

    df['data_saida'] = pd.to_datetime(df['data_saida'])
    group_by_week = df.groupby(pd.Grouper(key='data_saida', freq="W")).size().reset_index(name='count')
    group_by_week = group_by_week.iloc[1:-1]
    print("Agrupei")

    dataset_ready = dataset_preparation(group_by_week)

    X_train, X_test, y_train, y_test, test_dates = split_sequence(dataset_ready) 


    if file =="concelho_metropolitana_lisboa.csv":
        storage_url = "sqlite:///MLP_lisboa_sem_weekly.sqlite3"
    else:
        storage_url = "sqlite:///MLP_porto_sem_weekly.sqlite3"

 
    study_name = "Hidden Layers and Units Evaluation"
    #study_name = "Activation Function Evaluation"
    #study_name = "Optimizer Evaluation"
    #study_name = "Number Epochs Evaluation"

    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print("Study deleted successfully.")
    except KeyError:
        print("Study does not exist.")

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize')
    study.optimize(lambda trial: predict_hidden_units(trial, X_train, X_test, y_train, y_test, test_dates), n_trials=500)
    #study.optimize(lambda trial: predict_function(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=100)
    #study.optimize(lambda trial: predict_optimizer(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=20)
    #study.optimize(lambda trial: predict_epochs(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    