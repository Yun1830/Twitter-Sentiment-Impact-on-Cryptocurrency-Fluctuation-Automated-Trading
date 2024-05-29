import re
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 定義簡單的神經網路模型（DNN）
class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 文本預處理函數
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())  # 去除標點符號並轉換為小寫
    return text

# 封裝的預測函數
def predict_change_percentage_from_text_01(text):
    # 加载模型和其他必要的对象
    input_dim = 1003  #設定為1000維度+額外3特徵(sentiment,word_count,unique_word_ratio)
    model_dnn = DNNModel(input_dim)
    model_dnn.load_state_dict(torch.load('./models/Prediction_Josh_DNN/dnn_model_change_percentage.pth'))
    model_dnn.eval()

    scaler = torch.load('./models/Prediction_Josh_DNN/scaler.pth')
    tfidf = torch.load('./models/Prediction_Josh_DNN/tfidf.pth')
    sid = torch.load('./models/Prediction_Josh_DNN/sid.pth')

    # 進行預測的内部函數
    def predict_change_percentage(model, scaler, tfidf, sid, text):
        # 預處理文本
        preprocessed_text = preprocess_text(text)

        # 計算情感分析
        sentiment = sid.polarity_scores(preprocessed_text)['compound']

        # 計算其他文本特徵
        word_count = len(preprocessed_text.split())
        unique_word_ratio = len(set(preprocessed_text.split())) / len(preprocessed_text.split()) if len(preprocessed_text.split()) > 0 else 0

        # TF-IDF 特徵提取
        tfidf_features = tfidf.transform([preprocessed_text]).toarray()

        # 結合所有特徵
        features = np.hstack((tfidf_features, np.array([[sentiment, word_count, unique_word_ratio]])))

        # 標準化特徵
        features_scaled = scaler.transform(features)

        # 轉換為PyTorch張量
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # 進行預測
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)

        return prediction.item()

    # 輸入文本進行預測
    predicted_percentage = predict_change_percentage(model_dnn, scaler, tfidf, sid, text)
    return predicted_percentage

if __name__ == "__main__":
    # 如果直接運行此文件，進行以下操作
    text = input("請輸入文本以預測相對應的 Change_Percentage：")
    predicted_percentage = predict_change_percentage_from_text_01(text)
    print("預測的 Change_Percentage:", predicted_percentage)
