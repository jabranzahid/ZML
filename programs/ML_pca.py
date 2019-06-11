from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as scaler
%run ML_data
%run define_pca_model


data0 = ML_data(generate_random = True, nrandom_templates = 3000)
data1 = ML_data()
data2 = ML_data(evol_model = False)
data = [data0, data1, data2]

fpca, lpca = combine_ml_data(data)
f = fpca[0]
scaler().fit(f)
f = scaler.transform(f)

pca = PCA(0.99)
yfit=pca.fit(f)



data0 = ML_data(snr=0, generate_random = True, nrandom_templates = 3000)
data1 = ML_data()
data2 = ML_data(snr=10)
data4 = ML_data(evol_model = False)
data5 = ML_data(snr=10, evol_model = False)
data = [data0, data1, data2, data4, data5]
data = [data0, data1, data2, data4, data5]
features, labels = combine_ml_data(data)
ff = features[0]
ff = ss.transform(ff)
ff = pca.transform(ff)


model = define_pca_model(yfit.n_components_)
mmm = model.fit(ff, labels, batch_size = 128, epochs = 300,
                validation_split=0.05, verbose=1)




