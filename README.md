# Deep Recurrent Q-Network (DRQN) for Financial Data Analysis

This project implements a Deep Recurrent Q-Network (DRQN) to analyze financial data, specifically focusing on stock trading strategies using historical price data. The model employs reinforcement learning techniques to learn optimal trading actions based on market conditions.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)

## Installation

To set up the environment and run the DRQN model, follow these steps:

1. Clone the repository

## Dependencies

All dependencies of the project are within the projectSequential.py file itself.

## Usage

To run the DRQN model, execute the `projectSequential.py` script:

```bash
python projectSequential.py
```

The script will automatically download historical stock price data and perform training and testing of the DRQN model.

### Training

The model will train on historical stock data, learning to make decisions (Buy, Sell, Hold) based on market trends. The training process includes:

- A reduced number of training episodes for faster convergence.
- Experience replay to enhance learning stability.

### Testing

After training, the model will be tested over a set number of episodes. The test results will include the total profit/loss for each episode, which will be logged for review.


