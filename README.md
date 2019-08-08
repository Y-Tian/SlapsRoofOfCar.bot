# SlapsRoofOfCar.bot

**"Car Salesman: *slaps roof of car* this baby here can fit so many predictive models OwO"**

The objective of this bot is to predict the value/worth-it factor of buying a used car given the parameters: year, price, and mileage (of the used car).
By using neural networks with a dataset of roughly 200 slices, the bot will be able to tell you if the car is worth it based on current market trends for that
specific car.


**TLDR;** Found the perfect used car? Use this bot to check if it's actually a good deal. Goodluck!

*Note: location and prices only in the US, limitation of the dataset.



### Some examples of the model training & graphs of the results
Model training:
!(alt-text)[https://i.imgur.com/Y7hTZJR.png]
!(alt-text)[https://i.imgur.com/uzY72Jn.png]

Model Loss & Accuracy graphs:
!(alt-text)[https://i.imgur.com/qDunlIB.png]
!(alt-text)[https://i.imgur.com/rfybUdc.png]


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. Pulling datasets from `http://myslu.stlawu.edu/~clee/dataset/autotrader/`
2. Preprocessing this CSV dataset into (X, Y) `(210, 2) (45, 2) (45, 2) (210, 1) (45, 1) (45, 1)`
  * Training set
  * Validation set
  * Test set

### Installing

1. Requires the following packages to be installed onto your system
  * sklearn
  * pandas
  * numpy
  * keras
  * matplotlib

## Running the tests
1. (Optional)
  * Use venv: `source bin/activate`

2. To pull the dataset & preprocess the data
  * `python3 main.py` with the **flags** `--radius`, `--search_results`, `--without_csv`, `--dry_run`
  * Use `--help` if you don't know what the flags represent!
  
3. The script will prompt you to enter in the details of the car you're currently interested in
  * Ex. `kia`, `forte`, `2017`, `10.5` (thousand), `18.5` (thousand), `32703` (zipcode in Florida)

4. Give the model some time to evaluate. The result should be printed after a minute or so: `Under evaluation` or `Over evaluation`

## Agenda

1. Create a web interface to run the simulation
2. Reduce excessive deprecated logging from tensorflow

## Acknowledgments

* https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
