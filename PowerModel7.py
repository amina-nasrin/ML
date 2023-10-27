import pandas as pd
import numpy as np
import cpuinfo
import string
import psutil
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from os import cpu_count

my_cpu_info = cpuinfo.get_cpu_info()['brand_raw']
splited_cpu_info = my_cpu_info.split()
my_manufacturer_name = splited_cpu_info[0]
my_processor_model = splited_cpu_info[3]
my_number_of_cores = cpu_count()
my_cpu_load = psutil.cpu_percent(1)#for 10sec
df_my_row = {'processor_manufacturer': my_manufacturer_name,	'processor': my_processor_model,	'number_of_cores': 4,	'frequency': 2.45,	'load_percentile': my_cpu_load}
#df_my_power = {'power': 0}

df = pd.read_csv('230.csv')
x = df.drop(columns=['Platform', 'limitations', 'power', 'instance', 'memory_size(RAM)'])
x = x._append(df_my_row, ignore_index=True)#inserting my_processor into x

x['processor_manufacturer'] = x['processor_manufacturer'].astype('category')
x['processor_manufacturer'] = x['processor_manufacturer'].cat.codes

x['processor'] = x['processor'].astype('category')
x['processor'] = x['processor'].cat.codes

df['processor'] = df['processor'].astype('category')
df['processor'] = df['processor'].cat.codes

df['instance'] = df['instance'].astype('category')
df['instance_1'] = df['instance'].cat.codes

core_array = (df[['number_of_cores']]).astype(int).iloc[:,0].values
processor_manu_array = (df[['processor_manufacturer']]).to_string(index=False, header=False)

y = df[['power']]

x1 = x.tail(1)
#x = x.drop(x.index(x1))
last_row = len(x) - 1
x = x.drop(last_row)

print(x.head(5))
print(y.head(5))
poly_features = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly_features.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=.025, random_state=3)

model = LinearRegression()
model.fit(x_train, y_train)

x_vals = np.linspace(0, 100, 100).reshape(-1, 1)

y_pred_train = model.predict(x_train)
#plt.scatter(x_train, y_train)
#plt.plot(x, y_vals, color="r")
#plt.show()
#print(x_vals_poly, y_vals)

#df['Estimation'] = y_vals

df.to_csv('out_P7_all.csv', index = False)

y_pred_test = model.predict(x_test)

error = ((y_pred_test - y_test)/y_test)*100

#print('Testing data \n', y_test)
#print('\nPredicted data \n', y_pred_test)
#print(error)

mape = mean_absolute_percentage_error(y_test, y_pred_test)
#mse_test = mean_squared_error(y_test, y_pred_test)
print('\nMean Absolute % Error of Test Data ', mape*100, '%')

#df['Estimation'] = y_pred_test
#y_pred_this = model.predict()

print('\n\n\t\tMy Processor Specs')
print('----------------------------------------------------------------')
my_processor_row = x.loc[56,] #copying the specific row from the whole dataframe
my_processor_input = my_processor_row[['processor_manufacturer',	'processor',	'number_of_cores',	'frequency',	'load_percentile']]
my_processor_input = x1
print(my_processor_input)
my_processor_actual = y[['power']]

my_processor_input = np.asarray(my_processor_input)

my_processor_input = my_processor_input.reshape(1, -1)
my_processor_input_poly = poly_features.fit_transform(my_processor_input)

my_power = model.predict(my_processor_input_poly)
#my_mape = mean_absolute_percentage_error(my_processor_actual, my_power)
#print(my_power, my_processor_actual, '\nERROR ', my_mape*100, '%')
print('Predictied Power Consumption ', my_power)