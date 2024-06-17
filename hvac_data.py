import pandas as pd
pd.set_option('display.max_columns', None)

def get_hvac_data():
    dataset = pd.read_csv("simulation/Building01/data.csv", parse_dates=['Date'], index_col='Date')
    simulation_start = '2014-12-01'
    simulation_end = '2015-03-31'

    sub_data = dataset[(dataset.index >= simulation_start) & (dataset.index <= simulation_end)]
    print(sub_data.shape)

    return sub_data
    
get_hvac_data()