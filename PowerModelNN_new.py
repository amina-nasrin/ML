import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LeakyReLU, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.activations import relu, sigmoid
from keras.layers import LeakyReLU
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('230.csv')

processor_manu_array = (df[['processor_manufacturer']]).to_string(index=False, header=False)
df['processor_manufacturer'] = df['processor_manufacturer'].astype('category')
df['processor_manufacturer'] = df['processor_manufacturer'].cat.codes

df['processor'] = df['processor'].astype('category')
df['processor'] = df['processor'].cat.codes

x = df[['processor_manufacturer', 'number_of_cores', 'frequency', 'load_percentile']]
y = df[['power']]

y = np.asarray(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1)

#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_din = x_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='categorial_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(activation=sigmoid, layers=20, build_fn=create_model, verbose=1)

l = [[4], [40, 20]]#, [1]]
activations = ['sigmoid']#, 'relu']
param_grid = dict(layers=l, activation = activations, batch_size = [2, 6], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)

#print(grid_result)

