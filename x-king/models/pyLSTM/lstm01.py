import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords as nltk_stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import emoji
import nltk
# 設置環境變量以抑制TensorFlow日誌信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只顯示錯誤信息
#def load_trained_model(model_path='./models/pylstm/lstm.keras', info_path='./models/pylstm/lstm.pkl'):
nltk.download('stopwords')
def load_trained_model(model_path='./models/pyLSTM/lstm.keras', info_path='./models/pyLSTM/lstm.pkl'):
    # 加載模型
    model = load_model(model_path)
    
    # 加載其他信息
    with open(info_path, 'rb') as pkl_file:
        additional_info = pickle.load(pkl_file)
    
    tokenizer = additional_info['tokenizer']
    maxlen = additional_info['maxlen']
    
    return model, tokenizer, maxlen

# 加載訓練好的模型和 tokenizer
model, tokenizer, maxlen = load_trained_model()
# 初始化 TweetTokenizer 和 WordNetLemmatizer
tt = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk_stopwords.words('english'))

def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        # 將名詞、動詞和形容詞詞形還原
        lemma = lemmatizer.lemmatize(word, pos='n')  # 名詞
        lemma = lemmatizer.lemmatize(lemma, pos='v')  # 動詞
        lemma = lemmatizer.lemmatize(lemma, pos='a')  # 形容詞
        lemmatized_words.append(lemma)
        lemmatized_words = [re.sub(r'\d+', '', word) for word in lemmatized_words]
    return ' '.join(lemmatized_words)
    
def clean_tweet(tweet_text):
    # 轉換為小寫
    tweet = str(tweet_text)
    tweet = tweet.lower()
    # 使用 TweetTokenizer 分詞
    tweet = tt.tokenize(tweet)
    # 移除URL
    tweet = [re.sub(r'http\S+', '', i) for i in tweet]
    # 移除數字或特殊字符（保留emoji）
    tweet = [''.join(c for c in i if c not in punctuation or emoji.is_emoji(c)) for i in tweet]
    # 移除停用詞
    tweet = [i for i in tweet if i not in stopwords]
    return tweet
    
def predict_tweet(tweet):
    # 將新推文轉換為序列並填充
    sequence = tokenizer.texts_to_sequences([tweet])
    data = pad_sequences(sequence, maxlen=maxlen)
    # 使用訓練好的模型進行預測
    prediction = model.predict(data)
    prediction_float = prediction.item()
    formatted_float = round(prediction_float, 2)
    # rounded_prediction = [round(prob, 3) for prob in prediction[0]]
    # 返回數字列表而不是字符串
    return formatted_float

if __name__ == "__main__":
    # 允許用戶輸入字串進行預測
    while True:
        user_input = input("請輸入推文內容（或輸入'退出'以結束）：")
        if user_input.lower() == '退出':
            break
        cleaned = clean_tweet(user_input)
        lemmatized = lemmatize_words(cleaned)
        result = predict_tweet(lemmatized)
        print(result)
