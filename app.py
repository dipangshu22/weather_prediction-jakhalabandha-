import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score


df = pd.read_csv("weather_multi_months.csv")

df = df.iloc[::-1].reset_index(drop=True)

df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

df = df.dropna()




df["temp_prev"] = df["temp"].shift(1)


df["temp_next"] = df["temp"].shift(-1)
df["weather_next"] = df["weather"].shift(-1)

df = df.dropna()


X = df[["temp_prev", "humidity", "pressure"]]


y_temp = df["temp_next"]

split_index = int(len(df) * 0.8)

X_train_temp = X[:split_index]
X_test_temp = X[split_index:]

y_train_temp = y_temp[:split_index]
y_test_temp = y_temp[split_index:]

temp_model = LinearRegression()
temp_model.fit(X_train_temp, y_train_temp)

pred_temp = temp_model.predict(X_test_temp)

print("Temperature MAE:", mean_absolute_error(y_test_temp, pred_temp))


le = LabelEncoder()
df["weather_next_encoded"] = le.fit_transform(df["weather_next"])

y_weather = df["weather_next_encoded"]

X_train_w = X[:split_index]
X_test_w = X[split_index:]

y_train_w = y_weather[:split_index]
y_test_w = y_weather[split_index:]

weather_model = RandomForestClassifier(n_estimators=100, random_state=42)
weather_model.fit(X_train_w, y_train_w)

pred_weather = weather_model.predict(X_test_w)

print("Weather Accuracy:", accuracy_score(y_test_w, pred_weather))


latest = df.iloc[-1]

sample = pd.DataFrame([[
    latest["temp"],        
    latest["humidity"],
    latest["pressure"]
]], columns=["temp_prev", "humidity", "pressure"])

temp_pred = temp_model.predict(sample)[0]
weather_pred = weather_model.predict(sample)[0]

print("\nCurrent Temp:", latest["temp"])
print("Predicted Next Temp:", round(temp_pred, 2), "°C")
print("Predicted Next Weather:", le.inverse_transform([weather_pred])[0])