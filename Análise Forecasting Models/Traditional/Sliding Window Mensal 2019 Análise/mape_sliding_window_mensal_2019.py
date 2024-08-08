import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from pmdarima import auto_arima



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

def forecast_SES(train_data, increment):
    model = SimpleExpSmoothing(train_data)
    model_fit = model.fit()
    prediction = model_fit.forecast(increment)
    return prediction

def forecast_MA(train_data, increment):
    moving_avg_predictions = train_data.rolling(window=increment).mean()
    prediction = moving_avg_predictions.iloc[-1:]
    return prediction

def forecast_HW(train_data, increment, seasonal_periods):
    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    prediction = model_fit.forecast(steps=increment)
    return prediction

def forecast_ARIMA(train_data, increment):
    stepwise_fit = auto_arima(train_data, start_p=0, start_q=0, max_p=3, max_q=3, seasonal=False, suppress_warnings=True)
    prediction = stepwise_fit.predict(n_periods=increment)
    return prediction

def forecast_SARIMA(train_data, increment, seasonal_periods):
    stepwise_fit = auto_arima(train_data, start_p=0, start_q=0, max_p=3, max_q=3, m=seasonal_periods, seasonal=True, suppress_warnings=True)
    prediction = stepwise_fit.predict(n_periods=increment)
    return prediction


def predict(model, grouped_by):

    predictions = []
    mapes = []
    predictions_all = pd.DataFrame(columns = ['SES', 'MA', 'HW', 'ARIMA', 'SARIMA'])
    mape_all = pd.DataFrame(columns = ['SES', 'MA', 'HW', 'ARIMA', 'SARIMA'])


    if grouped_by == "M":
        begin = 0
        end = -12
        increment = 1
        seasonal_periods = 12

    while(True):

        if end == 0:
            break
    
        train_data = group_by_X["count"].iloc[begin:end]
        test_data = group_by_X["count"].iloc[end]

        if model == "SES":
            prediction = forecast_SES(train_data, increment)

        elif model == "MA":
            
            prediction = forecast_MA(train_data, increment)

        elif model == "HW":
            
            prediction = forecast_HW(train_data, increment, seasonal_periods)

        elif model == "ARIMA":
            
            prediction = forecast_ARIMA(train_data, increment)


        elif model == "SARIMA":

            prediction = forecast_SARIMA(train_data, increment, seasonal_periods)

        elif model == "ALL":
            prediction_SES = forecast_SES(train_data, increment)
            prediction_MA = forecast_MA(train_data, increment)
            prediction_HW = forecast_HW(train_data, increment, seasonal_periods)
            prediction_ARIMA = forecast_ARIMA(train_data, increment)
            prediction_SARIMA = forecast_SARIMA(train_data, increment, seasonal_periods)
            mape_SES = mape(test_data,prediction_SES)
            mape_MA = mape(test_data,prediction_MA)
            mape_HW = mape(test_data,prediction_HW)
            mape_ARIMA = mape(test_data,prediction_ARIMA)
            mape_SARIMA = mape(test_data,prediction_SARIMA)

            predictions_all = pd.concat([predictions_all, pd.DataFrame({'SES' : [prediction_SES.values[0]], 'MA' : [prediction_MA.values[0]], 'HW':[prediction_HW.values[0]], 'ARIMA': [prediction_ARIMA.values[0]], 'SARIMA' : [prediction_SARIMA.values[0]]})], ignore_index=True)
            mape_all = pd.concat([mape_all, pd.DataFrame({'SES' : [mape_SES], 'MA' : [mape_MA], 'HW':[mape_HW], 'ARIMA': [mape_ARIMA], 'SARIMA' : [mape_SARIMA]})], ignore_index=True)
            begin += increment
            end += increment
            continue

        
        predictions.append(prediction.values[0])
        mapes.append(mape(test_data,prediction))

        begin += increment
        end += increment
    if model == "ALL":
        return predictions_all, mape_all
    return predictions, mapes


def mape(test_data, prediction):
    
    mape = np.mean(np.abs((test_data - prediction) / test_data)) * 100
    return mape

def drawGraph(predictions, mapes, grouped_by_type, model, group_by_X, filename):

    if grouped_by_type == 'M':
        last_12_months = group_by_X['data_saida'].iloc[-12:]

        if model == 'ALL':
            # Iterate through columns of mapes DataFrame
            for column in mapes.columns:
                plt.plot(last_12_months, mapes[column], marker='o', linestyle='-', label=column)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Plot each column with label

        else:
            # Plot MAPE over time
            plt.plot(last_12_months, mapes, marker='o', linestyle='-')

        # Customize x-axis labels
        plt.xticks(last_12_months, last_12_months.dt.strftime('%b'))  # Display month abbreviations as labels

    plt.xlabel('Month')
    plt.ylabel('MAPE (%)')
    plt.title('Evolution of MAPE over Time ( '+ str(model) + ' ' + str(grouped_by_type) + ' )')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python combinacoes.py <file> <years> <groupby> <model>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: clean_consumos.csv or colheitas_10-05-2021.csv")
        print("- <years>: Number of years to filter the dataset, meaning it includes only data from the X most recent years.")
        print("- <groupby>: Type of grouping. Valid values: D or M or W")
        print("- <model>: Time Forecasting model. Valid values: SES or MA or HW or ARIMA or SARIMA or ALL")
        sys.exit(1)

    # Parse command-line arguments
    file = sys.argv[1]
    years = sys.argv[2]
    grouped_by_type = sys.argv[3]
    model = sys.argv[4]
    filename = f"{file}_{model}_{grouped_by_type}.png"

    df = readCSV(file)

    df_X_years = filterData(df, years)

    group_by_X = df_X_years.groupby(pd.Grouper(key='data_saida', freq=grouped_by_type)).size().reset_index(name='count')

    predictions, mapes = predict(model, grouped_by_type)

    drawGraph(predictions, mapes, grouped_by_type, model, group_by_X, filename)



# Começar em 4 de março




