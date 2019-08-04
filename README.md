# SlapsRoofOfCar.bot

**"Car Salesman: *slaps roof of car* this baby here can fit so many predictive models OwO"**

The objective of this bot is to predict the value/worth-it factor of buying a used car given the parameters: year, price, and mileage (of the used car).
By using neural networks with a dataset of roughly 200 slices, the bot will be able to tell you if the car is worth it based on current market trends for that
specific car.


**TLDR;** Found the perfect used car? Use this bot to check if it's actually a good deal. Goodluck!

*Note: location and prices only in the US, limitation of the dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. Pulling datasets from `http://myslu.stlawu.edu/~clee/dataset/autotrader/`
2. Preprocessing this CSV dataset into (X, Y) `(210, 2) (45, 2) (45, 2) (210, 1) (45, 1) (45, 1)`
  * Training set
  * Validation set
  * Test set

### Installing

1. Requires the following packages
  * sklearn
  * pandas
  * numpy
  * keras

## Running the tests

1. To pull the dataset & preprocess the data
  * `python3 main.py` with the **flags** `--car_make`, `--car_model`, `--zipcode`, `--radius`, `--search_results`, `--without_csv`
  * Use `--help` if you don't know what the flags represent!

## Agenda

1. Train the model with the processed dataset with keras!

## Acknowledgments

* https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
