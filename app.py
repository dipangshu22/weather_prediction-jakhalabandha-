import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score


df = pd.read_csv("weather_multi_months.csv")
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

df = df.dropna()


df["temp_prev"] = df["temp"].shift(1)
df = df.dropna()


X = df[["temp_prev", "humidity", "pressure"]]
y_temp = df["temp"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)

temp_model = LinearRegression()
temp_model.fit(X_train, y_train)

pred_temp = temp_model.predict(X_test)

print("Temperature MAE:", mean_absolute_error(y_test, pred_temp))

le = LabelEncoder()
df["weather_encoded"] = le.fit_transform(df["weather"])

y_weather = df["weather_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_weather, test_size=0.2, random_state=42
)

weather_model = RandomForestClassifier()
weather_model.fit(X_train, y_train)

pred_weather = weather_model.predict(X_test)

print("Weather Accuracy:", accuracy_score(y_test, pred_weather))



latest = df.iloc[-1]

sample = pd.DataFrame([[
    latest["temp"],
    latest["humidity"],
    latest["pressure"]
]], columns=["temp_prev", "humidity", "pressure"])

temp_pred = temp_model.predict(sample)[0]


weather_pred = weather_model.predict(sample)[0]

print("\nLast Temp:", latest["temp"])
print("Predicted Next Temp:", round(temp_pred, 2), "°C")
print("Predicted Weather:", le.inverse_transform([weather_pred])[0])