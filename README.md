import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Task 1: Fetch and Parse Data
url = "https://api.open-meteo.com/v1/forecast?latitude=12.97&longitude=77.59&hourly=temperature_2m,windspeed_10m"
response = requests.get(url).json()

# Parse hourly data into a dictionary
data = {
    "time": response["hourly"]["time"],
    "temperature_2m": response["hourly"]["temperature_2m"],
    "windspeed_10m": response["hourly"]["windspeed_10m"]
}

# Load into DataFrame
df = pd.DataFrame(data)
print("--- Task 1: Initial DataFrame (First 5 Rows) ---")
print(df.head())

# Task 2: Apply Feature Scaling
scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

# Apply MinMaxScaler to temperature_2m
df['temp_normalized'] = scaler_minmax.fit_transform(df[['temperature_2m']])

# Apply StandardScaler to windspeed_10m
df['wind_standardized'] = scaler_std.fit_transform(df[['windspeed_10m']])

print("\n--- Task 2: Scaled DataFrame (First 5 Rows) ---")
print(df.head())

--- Task 1: Initial DataFrame (First 5 Rows) ---
               time  temperature_2m  windspeed_10m
0  2026-04-08T00:00            21.2            1.0
1  2026-04-08T01:00            20.8            1.1
2  2026-04-08T02:00            21.8            0.4
3  2026-04-08T03:00            24.6            1.8
4  2026-04-08T04:00            27.1            0.4

--- Task 2: Scaled DataFrame (First 5 Rows) ---
               time  temperature_2m  windspeed_10m  temp_normalized  \
0  2026-04-08T00:00            21.2            1.0         0.023121   
1  2026-04-08T01:00            20.8            1.1         0.000000   
2  2026-04-08T02:00            21.8            0.4         0.057803   
3  2026-04-08T03:00            24.6            1.8         0.219653   
4  2026-04-08T04:00            27.1            0.4         0.364162   

   wind_standardized  
0          -1.721030  
1          -1.689803  
2          -1.908393  
3          -1.471212  
4          -1.908393  
