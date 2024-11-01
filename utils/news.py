import yfinance as yf

def get_apple_news():
    stock = yf.Ticker("AAPL")
    news_data = stock.news  # Get news related to Apple stock
    return news_data
