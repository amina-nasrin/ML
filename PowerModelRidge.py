import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv('2665.csv')

x = df[['load_percentile']]
y = df[['power_consumed']]

model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)

df['Estimation'] = y_pred

df.to_csv('out_Ridge_2665.csv', index = False)

mse = mean_squared_error(y, y_pred)
print(mse)