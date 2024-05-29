# Twitter Sentiment Impact on Cryptocurrency Fluctuation + Automated Trading

This project aims to analyze the impact of Twitter sentiment on the fluctuation of cryptocurrency prices and automatically execute trades based on the predictions. The system leverages deep learning models (DNN and LSTM) to predict price changes from tweet texts and makes automated trading decisions using Binance Futures Testnet.

## Features

1. **Twitter Sentiment Analysis**: 
   - Uses DNN and LSTM models to predict the percentage change in cryptocurrency prices based on tweet content.

2. **Automated Trading**:
   - Automatically places buy orders on the Binance Futures Testnet if the predicted price change exceeds a certain threshold.

3. **Real-Time Data Processing**:
   - Fetches current market prices and sets leverage and margin types for trades.

4. **User Interface**:
   - A Kivy-based GUI for inputting tweets, displaying predictions, and managing trading operations.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Binance API keys for the Futures Testnet
- Kivy for the GUI

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/<YOUR-GITHUB-USERNAME>/<REPOSITORY-NAME>.git
   cd <REPOSITORY-NAME>
