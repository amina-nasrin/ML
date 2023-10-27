import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv('2665.csv')

x = df[['load_percentile']]
y = df[['power_consumed']]

model = linear_model.LassoLars(alpha=.1)
model.fit(x, y)
y_pred = model.predict(x)

df['Estimation'] = y_pred

df.to_csv('out_PLassoLars_2665.csv', index = False)

mse = mean_squared_error(y, y_pred)
print(mse)