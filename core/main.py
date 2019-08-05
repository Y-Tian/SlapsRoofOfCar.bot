import click
import requests
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
import statistics
import matplotlib.pyplot as plt

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
@click.option('--dry_run', default=False, help='Without saving the keras model')
def start_core(car_make, car_model, zip_code, radius, search_results, without_csv, dry_run):
    if not without_csv:
        get_all_dataset(car_make, car_model, zip_code, radius, search_results)
    # preprocessing
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_all_preprocessing()
    # keras model
    model, hist = get_keras_model("", X_train, X_val, X_test, Y_train, Y_val, Y_test)
    # keras saving
    if not dry_run:
        save_keras_model(model, car_make, car_model)
    # visual plotting
    get_visual_plot(hist, "loss")
    get_visual_plot(hist, "accuracy")

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
    X = np.concatenate((dataset_x_prep_a, dataset_x_prep_b), 1)
    Y_temp = dataset[:,1:2]
    median_price = get_median_price(Y_temp)
    Y = format_input_dataset(Y_temp, median_price)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    Y_scale = min_max_scaler.fit_transform(Y)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y_scale, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def get_median_price(dataset):
    prices = []
    for item in dataset:
        for price in item:
            prices.append(price)

    return statistics.median(prices)

def compare_above_below_median(value, median):
    if value >= median:
        return 1
    return 0

def format_input_dataset(dataset, median):
    new_dataset = []
    for item in dataset:
        temp_dataset = []
        for price in item:
            temp_dataset.append(compare_above_below_median(price, median))
        new_dataset.append(temp_dataset)
    return new_dataset

def get_keras_model(model_type, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    # Basic model
    if model_type == "basic":
        model = Sequential([Dense(16, activation='relu', input_shape=(2,)), Dense(16, activation='relu'), Dense(1, activation='sigmoid'),])
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=14, epochs=100, validation_data=(X_val, Y_val))
        print(model.evaluate(X_test, Y_test)[1])

        return model, hist

    # Regularization model (does better because it reduces & eliminates overfitting)
    model_regularized = Sequential([Dense(750, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(2,)), Dropout(0.3), Dense(750, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(750, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),])
    model_regularized.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    hist_regularized = model_regularized.fit(X_train, Y_train, batch_size=16, epochs=100, validation_data=(X_val, Y_val))
    print(model_regularized.evaluate(X_test, Y_test)[1])

    return model_regularized, hist_regularized

def save_keras_model(model, car_make, car_model):
    model.save('%s_%s_model.h5' % (car_make, car_model))
    print('Keras model saved.')

def get_visual_plot(hist, plot_type):
    if plot_type == "loss":
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()
    elif plot_type == "accuracy":
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='lower right')
        plt.show()

if __name__ == '__main__':
    start_core()


# use the year and the mileage to predict the price
# need to revamp the dataset to include a field with "above median price or not"
# find median price of all datapoints, regardless of year and mileage

# HIDDEN LAYER NODES
# 𝑁ℎ=𝑁𝑠(𝛼∗(𝑁𝑖+𝑁𝑜))
# 𝑁𝑖  = number of input neurons.
# 𝑁𝑜 = number of output neurons.
# 𝑁𝑠 = number of samples in training data set.
# 𝛼 = an arbitrary scaling factor usually 2-10. (3 in the example case)