%run ML_data.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

pca, scaler = initialize_pca()
data0 = ML_data(snr=0, generate_random = True, nrandom_templates = 5000)
data1 = ML_data(snr=0)
data2 = ML_data(snr=10)
data3 = ML_data(snr=0, evol_model = False)
data4 = ML_data(snr=10, evol_model = False)
data5 = ML_data(snr=50)
data6 = ML_data(snr=50, evol_model = False)
data = [data0, data1, data2, data3, data4, data5, data6]
data = [data1, data3]
features, labels = combine_ml_data(data, pca = pca, scaler = scaler)
ff = features[0]

X_train, X_test, y_train, y_test = train_test_split(ff, labels, test_size=0.05, random_state=0)
regressor = RandomForestRegressor(n_estimators=400)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


test_data, mass = data[0].get_test_data()
test_pca = transform_data_pca(test_data[0], pca, scaler)
ttt = test_pca[0]
ppp = regressor.predict(ttt)

