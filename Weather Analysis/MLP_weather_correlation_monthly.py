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

def split_sequence(dataset_ready):
    reference_date = pd.Timestamp('2018-01-01')
    dataset_train = dataset_ready[dataset_ready['data_saida'] < reference_date]
    dataset_test = dataset_ready[dataset_ready['data_saida'] >= reference_date]

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
        model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(128, activation='relu'))
                # Hidden Layer 2
        model.add(Dense(64, activation='relu'))
                # Hidden Layer 3
        model.add(Dense(64, activation='relu'))
                # Hidden Layer 4
        model.add(Dense(112, activation='relu'))
                # Hidden Layer 5
        model.add(Dense(64, activation='relu'))
        print("LISBOA")
        print("LISBOA")
    else:
        model.add(Dense(72, activation="softmax", input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(120, activation="softmax"))
                # Hidden Layer 2
        model.add(Dense(120, activation="softmax"))
                # Hidden Layer 3
        model.add(Dense(48, activation="softmax"))

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
        # Hidden Layer 0
        model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(128, activation='relu'))
                # Hidden Layer 2
        model.add(Dense(64, activation='relu'))
                # Hidden Layer 3
        model.add(Dense(64, activation='relu'))
                # Hidden Layer 4
        model.add(Dense(112, activation='relu'))
                # Hidden Layer 5
        model.add(Dense(64, activation='relu'))
        print("LISBOA")
    else:
        model.add(Dense(72, activation="softmax", input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(120, activation="softmax"))
                # Hidden Layer 2
        model.add(Dense(120, activation="softmax"))
                # Hidden Layer 3
        model.add(Dense(48, activation="softmax"))

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
        model.add(Dense(16, activation=activation_layer, input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(128, activation=activation_layer))
                # Hidden Layer 2
        model.add(Dense(64, activation=activation_layer))
                # Hidden Layer 3
        model.add(Dense(64, activation=activation_layer))
                # Hidden Layer 4
        model.add(Dense(112, activation=activation_layer))
                # Hidden Layer 5
        model.add(Dense(64, activation=activation_layer))
        print("LISBOA")
    else:
        model.add(Dense(72, activation=activation_layer, input_dim=X_train.shape[1]))
        # Hidden Layer 1
        model.add(Dense(120, activation=activation_layer))
                # Hidden Layer 2
        model.add(Dense(120, activation=activation_layer))
                # Hidden Layer 3
        model.add(Dense(48, activation=activation_layer))

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
    if len(sys.argv) != 3:
        print("Usage: python MLP_without_weather.py <file>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: concelho_metropolitana_lisboa.csv or concelho_metropolitana_porto.csv")
        print("- <file2>: The name of the CSV file. Valid values: weather_lisboa.csv or weather_porto.csv")
        sys.exit(1)
    
    file = sys.argv[1]
    file2 = sys.argv[2]

    df = pd.read_csv(file)
    df2 = pd.read_csv(file2)
    df2.rename(columns={'date_time': 'data_saida'}, inplace=True)
    print("Li a data")

    df['data_saida'] = pd.to_datetime(df['data_saida'])
    df2['data_saida'] = pd.to_datetime(df2['data_saida'])
    group_by_month = df.groupby(pd.Grouper(key='data_saida', freq="M")).size().reset_index(name='count')
    print("Agrupei")

    pivot = pd.pivot_table(df2, 
                       index=pd.Grouper(key='data_saida', freq="M"),
                       values=['mintempC', 'maxtempC', 'precipMM', 'totalSnow_cm', 'sunHour', 'uvIndex', 'moon_illumination', 'DewPointC', 'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph', 'cloudcover', 'humidity', 'pressure', 'visibility', 'winddirDegree', 'windspeedKmph'], 
                       aggfunc={'mintempC': 'mean', 'maxtempC': 'mean', 'precipMM': 'sum', 'totalSnow_cm': 'sum', 'sunHour': 'mean', 'uvIndex': 'mean', 'moon_illumination': 'mean', 'DewPointC': 'mean', 'FeelsLikeC': 'mean', 'HeatIndexC': 'mean', 'WindChillC': 'mean', 'WindGustKmph': 'mean', 'cloudcover': 'mean', 'humidity': 'mean', 'pressure': 'mean', 'visibility': 'mean', 'winddirDegree': 'mean', 'windspeedKmph': 'mean'})


    merged_df = pd.merge(group_by_month, pivot, on='data_saida', how='left')
    print("Merged")

    X_train, X_test, y_train, y_test, test_dates = split_sequence(merged_df) 


    if file =="concelho_metropolitana_lisboa.csv":
        storage_url = "sqlite:///MLP_lisboa_CORRELATION_MONTHLY.sqlite3"
    else:
        storage_url = "sqlite:///MLP_porto_CORRELATION_MONTHLY.sqlite3"

 
    #study_name = "Hidden Layers and Units Evaluation"
    #study_name = "Activation Function Evaluation"
    #study_name = "Optimizer Evaluation"
    study_name = "Number Epochs Evaluation"

    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print("Study deleted successfully.")
    except KeyError:
        print("Study does not exist.")

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize')
    #study.optimize(lambda trial: predict_hidden_units(trial, X_train, X_test, y_train, y_test, test_dates), n_trials=500)
    #study.optimize(lambda trial: predict_function(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=100)
    #study.optimize(lambda trial: predict_optimizer(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=20)
    study.optimize(lambda trial: predict_epochs(trial, X_train, X_test, y_train, y_test, test_dates, file), n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    