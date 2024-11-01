from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from models.lstm_model import predict_lstm  # Assuming you have this model defined
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import smtplib
from email.mime.text import MIMEText
from utils.scraper import get_stock_data  # Your scraper module
import yfinance as yf

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Alert(BaseModel):
    email: str
    price: float

def send_email_alert(email: str, price: float):
    msg = MIMEText(f"Apple stock price has reached your alert level: ${price}")
    msg['Subject'] = 'Price Alert!'
    msg['From'] = "ronictakougang@gmail.com"  # Replace with your email
    msg['To'] = email

    with smtplib.SMTP('smtp-relay.sendinblue.com', 587) as server:  
        server.starttls()
        server.login("gunwaterco@gmail.com", "JmYtx390OGUBzWgn")  
        server.sendmail(msg['From'], [msg['To']], msg.as_string())

@app.post("/set-alert/")
async def set_alert(alert: Alert, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email_alert, alert.email, alert.price)
    return {"message": "Alert set!"}

@app.get("/price")
def get_price():
    prices = get_stock_data("AAPL")
    if prices:
        return {"prices": prices}  # Return all historical prices
    raise HTTPException(status_code=404, detail="Data not found")

@app.get("/predict")
def predict():
    prediction = predict_lstm()  # Assuming this is defined elsewhere
    return {"Prediction": prediction}

@app.get("/news")
def get_news():
    stock = yf.Ticker("AAPL")
    news_data = stock.news  # Get news related to Apple stock
    return {"news": news_data}

@app.get("/download-historical-data/")
def download_historical_data():
    prices = get_stock_data("AAPL")
    if not prices:
        raise HTTPException(status_code=404, detail="Historical data not found")

    # Create DataFrame with historical data
    stock = yf.Ticker("AAPL")
    historical_data = stock.history(period="3mo")
    historical_data.reset_index(inplace=True)  # Reset index to include Date as a column

    # Convert DataFrame to CSV
    csv = historical_data.to_csv(index=False)
    return StreamingResponse(io.StringIO(csv), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=historical_data.csv"})
