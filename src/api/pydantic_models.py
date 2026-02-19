 # src/api/pydantic_models.py
from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    CustomerId: int
    Amount_sum: float
    Amount_avg: float
    Amount_count: int
    Amount_std: float
    Value_sum: float
    Value_avg: float
    Value_count: int
    Value_std: float
    ProviderId_mode: int
    ProductId_mode: int
    ProductCategory_mode: int
    ChannelId_mode: int
    transaction_hour_avg: float
    transaction_day_avg: float
    transaction_month_avg: float
    transaction_year_avg: float
    transaction_dayofweek_avg: float
    is_weekend_avg: float
    is_business_hours_avg: float
    FraudResult: int
    
class RiskPredictionResponse(BaseModel):
    CustomerId: int
    risk_probability: float