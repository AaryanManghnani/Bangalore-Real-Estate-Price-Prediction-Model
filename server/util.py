import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print('Loading saved artifacts...Start')
    global __data_columns
    global __locations
    global __model
    
    with open("C:\\Users\\aarya\\OneDrive\\Desktop\\Python Projects\\ML Projects\\Bangalore Real Estate Price Prediction Project\\server\\artifacts\\columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    if __model is None:
        with open("C:\\Users\\aarya\\OneDrive\\Desktop\\Python Projects\\ML Projects\\Bangalore Real Estate Price Prediction Project\\server\\artifacts\\price_prediction_model.pickle",'rb') as f:
            __model = pickle.load(f)
    print("Loading saved artifacts...Done")
    
def get_location_names():
    return __locations
    
def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print (get_location_names()) 