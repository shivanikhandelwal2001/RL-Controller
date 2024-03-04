import pandas as pd

def get_hvac_data():

    data = pd.read_csv("hvac_data.csv", parse_dates=['Date'])
    gas_data = pd.read_pickle("Building01_Processed.pkl")[["Gas_volume"]].reset_index()
    
    data = pd.merge(data, gas_data, on='Date', how='inner')
    data.to_csv("hvac_gas_data.csv", index=False)

    spaces_with_temp = set(col.split('_')[0] for col in data.columns if 'Air_temperature_pred' in col)
    spaces_with_humidity = set(col.split('_')[0] for col in data.columns if 'Relative_humidity_pred' in col)

    # Finding spaces with both temperature and humidity data
    spaces_with_both = spaces_with_temp.intersection(spaces_with_humidity)

    # Filtering columns for those spaces
    filtered_columns = [col for col in data.columns if col.split('_')[0] in spaces_with_both]

    temp_columns = sorted([col for col in filtered_columns if 'Air_temperature_pred' in col])
    humidity_columns = sorted([col for col in filtered_columns if 'Relative_humidity_pred' in col])
    paired_columns = [val for pair in zip(temp_columns, humidity_columns) for val in pair]
    data1 = data[paired_columns]

    return data1, data[["Gas_volume"]]