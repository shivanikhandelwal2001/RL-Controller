import pandas as pd
pd.set_option('display.max_columns', None)

def get_hvac_data():
    dataset = pd.read_pickle("Building01_Processed.pkl")
    
    # predictions = pd.read_csv("Building01_LinearRegression_predictions.csv")
    # print(predictions.head())
    # print(predictions.shape)
    # print(predictions.columns)

    # spaces_with_temp = set(col.split('_')[0] for col in data.columns if 'Air_temperature_pred' in col)
    # spaces_with_humidity = set(col.split('_')[0] for col in data.columns if 'Relative_humidity_pred' in col)

    # # Finding spaces with both temperature and humidity data
    # spaces_with_both = spaces_with_temp.intersection(spaces_with_humidity)

    # # Filtering columns for those spaces
    # filtered_columns = [col for col in data.columns if col.split('_')[0] in spaces_with_both]

    # temp_columns = sorted([col for col in filtered_columns if 'Air_temperature_pred' in col])
    # humidity_columns = sorted([col for col in filtered_columns if 'Relative_humidity_pred' in col])
    # paired_columns = [val for pair in zip(temp_columns, humidity_columns) for val in pair]
    # data1 = data[paired_columns]

    return dataset
    
get_hvac_data()