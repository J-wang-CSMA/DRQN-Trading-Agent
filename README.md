# Deep Recurrent Q-Network (DRQN) for Financial Data Analysis

This project implements a Deep Recurrent Q-Network (DRQN) to analyze financial data, specifically focusing on stock trading strategies using historical price data. The model employs reinforcement learning techniques to learn optimal trading actions based on market conditions.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [Logging](#logging)
- [License](#license)

## Installation

To set up the environment and run the DRQN model, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Yuvaraj0702/3263research.git
   cd 3263research
   ```

## Dependencies

This project requires the following Python packages:

- `numpy`
- `pandas`
- `gym`
- `yfinance`
- `tensorflow`
- `logging`

You can install them using pip:
```bash
pip install numpy pandas gym yfinance tensorflow
```

## Usage

To run the DRQN model, execute the `project.py` script:

```bash
python project.py
```

The script will automatically download historical stock price data (default: Apple Inc. - AAPL) and perform training and testing of the DRQN model.

### Training

The model will train on historical stock data, learning to make decisions (Buy, Sell, Hold) based on market trends. The training process includes:

- A reduced number of training episodes for faster convergence.
- Experience replay to enhance learning stability.

### Testing

After training, the model will be tested over a set number of episodes. The test results will include the total profit/loss for each episode, which will be logged for review.

## Logging

The application uses Python's built-in logging module to provide real-time insights into the training and testing processes. The log will detail:

- Episode start and completion.
- Actions taken during testing.
- Total rewards for each episode.

You can adjust the logging level in the code to show more or less detail.

