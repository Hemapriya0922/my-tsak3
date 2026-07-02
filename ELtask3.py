import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the dataset
df = pd.read_csv(r"C:\Users\hemap\Downloads\Housing.csv")

# Display first five rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Convert yes/no columns into 1/0
yes_no_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in yes_no_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Convert furnishingstatus into dummy variables
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

print(df.head())
# Features (X)
X = df.drop('price', axis=1)

# Target (y)
y = df['price']

print("Features:")
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Set:", X_train.shape)
print("Testing Set:", X_test.shape)
#sklearn.linear_model
# Import Linear Regression
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train (fit) the model using the training data
model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")
y_pred = model.predict(X_test)

print("Predicted Prices:")
print(y_pred[:10])
#MAE, MSE, R²
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)

print("\nIntercept:", model.intercept_)
print("Model Evaluation")
print("-----------------------")
print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("R² Score            :", r2)

print("\nFeature Coefficients")
print(coefficients)

print("\nIntercept:", model.intercept_)
