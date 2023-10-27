import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error

df = pd.read_csv('new.csv')
x = df[['100',	'90',	'80',	'70',	'60',	'50',	'40',	'30',	'20', '10']]
y = df[['0']]

x = np.asarray(x)
y = np.asarray(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=4)

y = y_train.ravel()
y_train = np.asarray(y).astype(int)

#print(y_train)
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
classifier.fit(x_train, y_train)
#classifier.fit(x_train, y_train.values.reshape(-1,))

y_predict = classifier.predict(x_test)
print('Testing Data\n', y_test)
print('Predicted Data\n', y_predict)

mse = mean_squared_error(y_test, y_predict)
print('Mean Squared Error ', mse)