
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

- Python 3.9
- Binance API keys for the Futures Testnet
- Kivy for the GUI

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/<YOUR-GITHUB-USERNAME>/<REPOSITORY-NAME>.git
   cd <REPOSITORY-NAME>
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Configuration

Set your Binance API key and secret in the script:
   ```python
   api_key = 'your_api_key'
   api_secret = 'your_api_secret'
   ```

### Running the Application

   ```sh
   python combination_v6-1.py
   ```

### Usage

- **Input a Tweet**: Enter the text of a tweet in the provided input field.
- **Predict and Trade**: The system will predict the impact on the cryptocurrency price and automatically place an order if the prediction meets the criteria.
- **View Results**: See the prediction details and order execution information in the GUI.

### Contributions

Contributions are welcome. Please submit a pull request or open an issue to discuss improvements or bug fixes.
