import pandas as pd

pd.set_option("display.max_columns", None)


def get_hvac_data():
    dataset = pd.read_pickle("simulation/BuildingA/Building01_Processed.pkl")
    simulation_start = "2014-12-01"
    simulation_end = "2015-03-31"

    sub_data = dataset[
        (dataset.index >= simulation_start) & (dataset.index <= simulation_end)
    ]
    
    sub_data.reset_index(inplace=True, drop=True)
    print(sub_data.shape)

    return sub_data


get_hvac_data()
