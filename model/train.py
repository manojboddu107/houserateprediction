import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#loading the dataset
data = pd.read_csv('data/houserate.csv')

#input data
x = data[['area', 'bed_rooms', 'age']]
y = data['price']

#create model
model = LinearRegression()

#train the model with input features and output variable
model.fit(x, y)

#display learned parameters
print("slope-m(weight) =", model.coef_)
print("intercept - c (constant) =", model.intercept_)

#test prediction
sq_feet = [[1500, 3, 5]]
pred_price = model.predict(sq_feet)
print("Predicted price = ", pred_price[0])

#save the model
joblib.dump(model, 'model/house_price_model.pkl')
print("Model saved successfully")