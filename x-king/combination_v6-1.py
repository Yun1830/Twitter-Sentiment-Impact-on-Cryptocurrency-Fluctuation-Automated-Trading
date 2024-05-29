# version 6
# 計算判斷與下單花費時間
# 固定下單數量與槓桿
# 每次按下按鈕清除前一比資料
# call myapp.kv


from binance.client import Client
from binance.enums import *
import json
import time

# 設置Binance Futures Testnet的API Key和Secret Key
api_key = 'your_api_key'
api_secret = 'your_api_secret'

# Futures Testnet終端URL
futures_testnet_url = 'https://testnet.binancefuture.com'

# 創建客戶端，並設置Futures Testnet的終端URL
client = Client(api_key, api_secret, testnet=True)
client.FUTURES_API_URL = futures_testnet_url

def place_order(symbol='BTCUSDT', leverage=100, quantity=0.002, margin_type='ISOLATED'):
    try:
        start_time = time.time()

        # 獲取當前市場價格
        avg_price = client.futures_mark_price(symbol=symbol)
        current_price = float(avg_price['markPrice'])
        print(f"當前價格: {current_price} USDT")

        # 檢查並設置保證金類型
        try:
            response = client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            print(f"保證金類型設置為: {margin_type}，回應: ", response)
        except Exception as e:
            if "No need to change margin type" in str(e):
                print(f"保證金類型已經是 {margin_type}，無需更改")
            else:
                raise e

        # 設置槓桿
        response = client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"槓桿設置為: {leverage} 倍，回應: ", response)

        # 創建市場買單
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity
        )

        # 印出成交信息
        filled_amount = float(order["cumQuote"])
        order_info = {
            "orderId": order["orderId"],
            "symbol": order["symbol"],
            "side": order["side"],
            "origType": order["type"],
            "origQty": order["origQty"],
            "filledAmount": filled_amount
        }
        print("成交資訊")
        print(json.dumps(order_info, indent=4))

        # 計算時間
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"下單耗時: {time_taken:.2f} 秒")

        return "下單成功", order_info, time_taken

    except ValueError as ve:
        print("操作失敗", ve)
        return "操作失敗", str(ve), 0
    except Exception as e:
        print("操作失敗", e)
        return "操作失敗", str(e), 0

from kivy.config import Config
Config.set('kivy', 'default_font', [
    '微軟正黑體-1',
    './微軟正黑體-1.ttf'
])
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from models.Prediction_Josh_DNN.predict_change import predict_change_percentage_from_text_01
from models.pyLSTM.lstm01 import predict_tweet

# 設置初始窗口大小和最小窗口大小
Window.size = (800, 600)
Window.minimum_width = 800 
Window.minimum_height = 600 

class JudgmentScreen(Screen):
    def analyze_tweet_01(self, tweet):
        try:
            predicted_percentage = predict_change_percentage_from_text_01(tweet)
            return predicted_percentage
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

    def analyze_tweet_lstm(self, tweet):
        try:
            prediction = predict_tweet(tweet)
            return prediction
        except Exception as e:
            print(f"An error occurred during LSTM prediction: {e}")
            return None

    def clear_input(self):
        self.ids.tweet_input.text = ''
        self.ids.lstm_result_label.text = ''
        self.ids.dnn_result_label.text = ''
        self.ids.avg_result_label.text = ''
        self.ids.order_result_label.text = ''
        self.ids.order_info_label.text = ''
        self.ids.time_taken_label.text = ''

    def judge_tweet(self):
        tweet = self.ids.tweet_input.text
        dnn_result = self.analyze_tweet_01(tweet)
        lstm_result = self.analyze_tweet_lstm(tweet)
        
        if dnn_result is not None and lstm_result is not None:
            avg_result = (dnn_result + lstm_result) / 2
        else:
            avg_result = None

        if dnn_result is not None:
            self.ids.dnn_result_label.text = f'DNN預測值: {dnn_result:.2f}%'

        if lstm_result is not None:
            self.ids.lstm_result_label.text = f'LSTM預測值: {lstm_result:.2f}%'

        if avg_result is not None:
            self.ids.avg_result_label.text = f'模型最終預測值: {avg_result:.2f}%'
            if avg_result < 0.8:
                self.ids.order_result_label.text = "預測結果過低，不予下單"
                self.ids.order_result_box.opacity = 1  # 顯示此消息
                self.ids.order_info_label.text = ""
                self.ids.time_taken_label.text = ""
            else:
                order_result, order_info, time_taken = place_order(leverage=100, quantity=0.002, margin_type='ISOLATED')
                self.ids.order_result_label.text = order_result
                self.ids.order_info_label.text = json.dumps(order_info, indent=4, ensure_ascii=False)
                self.ids.time_taken_label.text = f'下單耗時: {time_taken:.2f} 秒'
                self.ids.order_result_box.opacity = 1  # 顯示

        display_text = tweet if len(tweet) <= 30 else tweet[:27] + '...'
        history_button = Button(text=display_text, size_hint_y=None, height='48dp', on_press=self.reuse_tweet)
        history_button.full_text = tweet  # 將完整文字保存在按鈕屬性中
        self.ids.history_box.add_widget(history_button)

    def clear_history(self):
        self.ids.history_box.clear_widgets()

    def reuse_tweet(self, instance):
        self.ids.tweet_input.text = instance.full_text  # 使用完整文字

class MyApp(App):
    def build(self):
        self.title = 'X-KING'
        sm = ScreenManager()
        sm.add_widget(JudgmentScreen(name='judgment'))
        return sm

if __name__ == '__main__':
    MyApp().run()