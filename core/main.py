import click
import requests
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

autotrader_url = 'http://myslu.stlawu.edu/~clee/dataset/autotrader/retrieve.php?'
csv_file = 'data.csv'

@click.command()
# @click.option('--car_make', prompt='Car make', help='The brand of the car')
# @click.option('--car_model', prompt='Car model', help='The model of the car')
# @click.option('--zip_code', prompt='Zip code', help='Your zip code')
@click.option('--car_make', default='kia', help='The brand of the car')
@click.option('--car_model', default='forte', help='The model of the car')
@click.option('--zip_code', default=32703, help='Your zip code')
@click.option('--radius', default=100, help='Radius of car searches with respect to zip code')
@click.option('--search_results', default=300, help='Amount of search results')
@click.option('--without_csv', default=True, help='If you already have a csv ready')
def start_core(car_make, car_model, zip_code, radius, search_results, without_csv):
    if not without_csv:
        get_all_dataset(car_make, car_model, zip_code, radius, search_results)
    get_all_preprocessing()

def get_all_dataset(car_make, car_model, zip_code, radius, search_results):
    click.echo('Grabbing data from Autotrader.com for %s %s at location %s' % (car_make, car_model, zip_code))
    tuple_of_car_info = (car_make, car_model, zip_code, radius, search_results)
    # write the csv into a local file
    get_autotrader_data(tuple_of_car_info)

def get_autotrader_data(tuple_of_car_info):
    temp_autotrader_url = autotrader_url
    temp_autotrader_url += ('make=%s' % tuple_of_car_info[0].upper())
    temp_autotrader_url += ('&model=%s' % tuple_of_car_info[1].upper())
    temp_autotrader_url += ('&zipcode=%s' % tuple_of_car_info[2])
    temp_autotrader_url += ('&radius=%s' % tuple_of_car_info[3])
    temp_autotrader_url += ('&limit=%s' % tuple_of_car_info[4])
    resp = requests.get(temp_autotrader_url)
    data_text = resp.text
    with open(csv_file, 'w') as f:
        f.write(data_text)

def get_all_preprocessing():
    df = pd.read_csv(csv_file)
    dataset = df.values
    dataset_x_prep_a = np.array(dataset[:,:1])
    dataset_x_prep_b = np.array(dataset[:,2:])
    X = np.concatenate((dataset_x_prep_a, dataset_x_prep_a), 1)
    Y = dataset[:,1:2]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    Y_scale = min_max_scaler.fit_transform(Y)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y_scale, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

if __name__ == '__main__':
    start_core()


# use the year and the mileage to predict the price