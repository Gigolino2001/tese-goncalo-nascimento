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

def predict(group_by_Day,days, model):


    
    main_columns = ['SES', 'MA', 'HW', 'ARIMA', 'SARIMA']
    #main_columns = ['SES', 'MA', 'HW']


    # Create a list of subcolumn names
    sub_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_dict = {day: i for i, day in enumerate(sub_columns)}

    # Create a MultiIndex for the columns
    columns = pd.MultiIndex.from_product([main_columns, sub_columns])

    # Create an empty DataFrame with the multi-index columns
    predictions_all = pd.DataFrame(columns=columns)
    mape_all = pd.DataFrame(columns =columns)
    predictions = pd.DataFrame(columns=sub_columns)
    mapes = pd.DataFrame(columns=sub_columns)
    seasonal_periods = 7

    for days_of_week in week_dict:
        train_data, test_data = filterExactDate(group_by_Day, week_dict[days_of_week])
        print("FilterExactDate " + str(days_of_week))

        pred_temp = []
        mape_temp = []

        pred_temp_SES = []
        pred_temp_MA = []
        pred_temp_HW = []
        pred_temp_ARIMA = []
        pred_temp_SARIMA = []
        mape_temp_SES = []
        mape_temp_MA = []
        mape_temp_HW = []
        mape_temp_ARIMA = []
        mape_temp_SARIMA = []

        for day in days:
            


            test_data_local = test_data.iloc[:day]

            if model == "SES":
                prediction = forecast_SES(train_data, day)

            elif model == "MA":
                
                prediction = forecast_MA(train_data, day)

            elif model == "HW":
                
                prediction = forecast_HW(train_data, day, seasonal_periods)

            elif model == "ARIMA":
                
                prediction = forecast_ARIMA(train_data, day)


            elif model == "SARIMA":

                prediction = forecast_SARIMA(train_data, day, seasonal_periods)

            elif model == "ALL":
                prediction_SES = forecast_SES(train_data, day)
                prediction_MA = forecast_MA(train_data, day)
                prediction_HW = forecast_HW(train_data, day, seasonal_periods)
                #prediction_ARIMA = forecast_ARIMA(train_data, day)
                #prediction_SARIMA = forecast_SARIMA(train_data, day, seasonal_periods)
                mape_SES = mape(test_data_local,prediction_SES, "SES")
                mape_MA = mape(test_data_local,prediction_MA, "MA")
                mape_HW = mape(test_data_local,prediction_HW, "HW")
                #mape_ARIMA = mape(test_data_local,prediction_ARIMA, "ARIMA")
                #mape_SARIMA = mape(test_data_local,prediction_SARIMA, "SARIMA")


                pred_temp_SES.append(prediction_SES.values[0])
                pred_temp_MA.append(prediction_MA.values[0])
                pred_temp_HW.append(prediction_HW.values[0])
                #pred_temp_ARIMA.append(prediction_ARIMA.values[0])
                #pred_temp_SARIMA.append(prediction_SARIMA.values[0])
                mape_temp_SES.append(mape_SES)
                mape_temp_MA.append(mape_MA)
                mape_temp_HW.append(mape_HW)
                #mape_temp_ARIMA.append(mape_ARIMA)
                #mape_temp_SARIMA.append(mape_SARIMA)
                continue
               
            pred_temp.append(prediction.values[0])
            mape_temp.append(mape(test_data_local,prediction, model))
        if model == 'ALL':
            predictions_all[('SES',days_of_week)] = pred_temp_SES
            predictions_all[('MA',days_of_week)] = pred_temp_MA
            predictions_all[('HW',days_of_week)] = pred_temp_HW
            #predictions_all[('ARIMA',days_of_week)] = pred_temp_ARIMA
            #predictions_all[('SARIMA',days_of_week)] = pred_temp_SARIMA
            mape_all[('SES',days_of_week)] = mape_temp_SES
            mape_all[('MA',days_of_week)] = mape_temp_MA
            mape_all[('HW',days_of_week)] = mape_temp_HW
            #mape_all[('ARIMA',days_of_week)] = mape_temp_ARIMA
            #mape_all[('SARIMA',days_of_week)] = mape_temp_SARIMA      
        else:
            predictions[days_of_week] = pred_temp
            mapes[days_of_week] = mape_temp

    if model == "ALL":
        return predictions_all, mape_all
    return predictions, mapes

def mape(test_data, prediction, model):
    length = test_data.shape[0]
    if model == "MA":
        test_data = test_data.reset_index()
        prediction = prediction.reset_index()
        #sum_test_data = test_data['count'].sum() 
        #sum_prediction = prediction['count'].sum() * length
    #else:
        #sum_test_data = test_data.sum()
        #sum_prediction = prediction.sum()
    
    #subtract = (sum_test_data - sum_prediction)
    #tmp_mape = np.mean(np.abs((subtract) / sum_test_data)) * 100
    #mape = tmp_mape / length
        
    subtract = (test_data - prediction)
    mape = np.mean(np.abs((subtract) / test_data)) * 100
    return mape

def calculate_average_mapes(mapes):
    sub_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    #main_columns = ['SES', 'MA', 'HW', 'ARIMA', 'SARIMA']
    main_columns = ['SES', 'MA', 'HW']
    average_mape = pd.DataFrame(columns=main_columns)
    for model in main_columns:
        sum_mapes = np.zeros(14)
        for day_week in sub_columns:
            sum_mapes = np.add(sum_mapes, mapes.loc[:, (model, day_week)].values)
        average_mape[model] = (sum_mapes / len(sub_columns)).tolist()

    return average_mape



def drawGraph(predictions, mapes, model, filename):
    
    days = ["1","2","3","4","5","6","1 week", "2 week", "3 week", "4 week", "5 week", "6 week", "7 week", "8 week"]

    if model == 'ALL':
        replacement_dict = {np.inf: 1e10}
        mapes = mapes.replace(replacement_dict)
        mapes = calculate_average_mapes(mapes)
        # Iterate through columns of mapes DataFrame
        for column in mapes.columns:
            plt.plot(days, mapes[column], marker='o', linestyle='-', label=column)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Plot each column with label

    else:
        replacement_dict = {np.inf: 1e10}
        mapes = mapes.replace(replacement_dict)
        # Iterate through columns of mapes DataFrame
        for column in mapes.columns:
            plt.plot(days, mapes[column], marker='o', linestyle='-', label=column)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


    # Set y-axis limit to 300
    plt.ylim(0, 100)    
    plt.xlabel('Period')
    plt.ylabel('MAPE (%)')
    plt.title(f'Granularity performance of model {model} in dataset {filename} by Day of the Week')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename)

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python combinacoes.py <file> <years> <groupby> <model>")
        print("Valid arguments:")
        print("- <file>: The name of the CSV file. Valid values: clean_consumos.csv or colheitas_10-05-2021.csv")
        print("- <model>: Time Forecasting model. Valid values: SES or MA or HW or ARIMA or SARIMA or ALL")
        sys.exit(1)

    # Parse command-line arguments
    file = sys.argv[1]
    model = sys.argv[2]
    years = 10
    filename = f"{file}_{model}_granularidade_dias_semana.png"
    days = [1,2,3,4,5,6,7,14,21,28,35,42,49,56]
    df = readCSV(file)

    print("Li a data")

    df_X_years = filterData(df, years)
    print("Filtrei a data")

    group_by_Day = df_X_years.groupby(pd.Grouper(key='data_saida', freq="D")).size().reset_index(name='count') 
    print("Agrupei")

    predictions, mapes = predict(group_by_Day, days, model)
    print("Dei predict")


    """ file2 = 'mapes_all clean_consumos.csv.csv'
    file3 = 'predictions_all clean_consumos.csv.csv'

    mapes = pd.read_csv(file2,  header=[0, 1], index_col=0)
    predictions = pd.read_csv(file2, header=[0,1], index_col = 0) """

    drawGraph(predictions, mapes, model, filename)

    print("Acabei")

