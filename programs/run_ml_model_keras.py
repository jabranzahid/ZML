%run ML_data.py
%run define_cnn_model.py
%run define_dense_model.py
%run define_pca_model.py
%run define_split_model.py
%run define_split_model_plus.py
%run define_split_model_plus_ebv.py
%run fit_model_plot_loop.py


#for keras model with 8, 256, 128, 10 layers
#this model saved on desktop as keras_model2.h5
n_chunks = 25
data1 = ML_data(snr=0, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 1)
data2 = ML_data(snr=5, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data3 = ML_data(snr=10, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data4 = ML_data(snr=20, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data5 = ML_data(snr=50, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data = [data1, data2, data3, data4, data5]
model = define_split_model_plus(data[0], l1 = 1e-5, l2 = 1e-5)
data_temp = ML_data(snr=0, n_chunks = n_chunks, evol_model = False, validation_split=0.02)
model, pout = fit_model_plot_loop(model, [data_temp], n_loops = 100, plot_ind_spec = False)
model, pout = fit_model_plot_loop(model, data, n_loops = 2501, plot_ind_spec = True)
#finish with SGD rather than ADAM
model.compile(loss='mean_squared_error', optimizer='sgd')
model, pout = fit_model_plot_loop(model, data, n_loops = 500, plot_ind_spec = True)


#for keras model with 32, 256, 128, 64, 10 layers
n_chunks = 25
data1 = ML_data(snr=0, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 1)
data2 = ML_data(snr=5, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data3 = ML_data(snr=10, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data4 = ML_data(snr=20, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data5 = ML_data(snr=50, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.25)
data = [data1, data2, data3, data4, data5]
data_temp = ML_data(snr=0, n_chunks = n_chunks, evol_model = True, validation_split=0.02)
model = define_split_model_plus(data_temp, l1 = 1e-7, l2 = 1e-7)
model, pout = fit_model_plot_loop(model, [data_temp], n_loops = 35, plot_ind_spec = False)
model, pout = fit_model_plot_loop(model, data, n_loops = 20001, plot_ind_spec = True)



#I-Ting run these lines for starters
n_chunks = 25
data1 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False, const_sfr = True, validation_split=0.02)
data2 = ML_data(n_chunks = n_chunks, snr=10, evol_model = False, const_sfr = True, validation_split=0.02)
data = [data1, data2]
model = define_split_model_plus(data[0], l1 = 1e-5, l2 = 1e-5)
model, pout = fit_model_plot_loop(model, data, n_loops = 20001, plot_ind_spec = True)






n_chunks = 25
data1 = ML_data(snr=0, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.02)
data2 = ML_data(snr=5, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.005)
data3 = ML_data(snr=10, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.005)
data4 = ML_data(snr=20, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.005)
data5 = ML_data(snr=50, n_chunks = n_chunks, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.005)
data = [data1, data2, data3, data4, data5]
model = define_split_model_plus(data[0], l1 = 1e-5, l2 = 1e-5)
model, pout = fit_model_plot_loop(model, data, n_loops = 20001, plot_ind_spec = False)




pca, scaler = initialize_pca()
data1 = ML_data(snr=0, evol_model = False, const_sfr = True, validation_split=0.02, training_data_fraction = 0.2)
data2 = ML_data(snr=5, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.05)
data3 = ML_data(snr=10, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.05)
data4 = ML_data(snr=20, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.05)
data5 = ML_data(snr=50, evol_model = False, const_sfr = True, validation_split=0.005, training_data_fraction = 0.05)
data = [data1, data2, data3, data4, data5]
model = define_pca_model(pca.n_components_, l1 = 1e-3, l2 = 1e-3)
model, pout = fit_model_plot_loop(model, data, n_loops = 1001, pca = pca, scaler = scaler)



pca, scaler = initialize_pca()
data1 = ML_data(snr=0)
data2 = ML_data(snr=20)
data3 = ML_data(snr=0, evol_model = False)
data4 = ML_data(snr=20, evol_model = False)
data5 = ML_data(snr=0, generate_random = True, nrandom_templates = 1000, n_random_avg = 3)
data6 = ML_data(snr=20, generate_random = True, nrandom_templates = 1000, n_random_avg = 3)
data = [data1, data2, data3, data4, data5, data6]
model = define_pca_model(pca.n_components_, l1 = 1e-3, l2 = 1e-3)
model, pout = fit_model_plot_loop(model, data, n_loops = 1001, pca = pca, scaler = scaler)


pca, scaler = initialize_pca()
data0 = ML_data(snr=0, generate_random = True, nrandom_templates = 300, n_random_avg = 3)
data1 = ML_data(snr=25, generate_random = True, nrandom_templates = 300, n_random_avg = 3)
model = define_pca_model(pca.n_components_, l1 = 0, l2 = 0)
model, pout = fit_model_plot_loop(model, [data0, data1], n_loops = 10001, pca = pca, scaler = scaler, plot_ind_spec=False)



#PCA applroach under heavy development
pca, scaler = initialize_pca()
data0 = ML_data(snr=0, generate_random = True, nrandom_templates = 5000)
data1 = ML_data(snr=10, generate_random = True, nrandom_templates = 5000)
data = [data0, data1]
model = define_pca_model(pca.n_components_, l1 = 1e-3, l2 = 1e-3)
model, pout = fit_model_plot_loop(model, data, n_loops = 1001, pca = pca, scaler = scaler, plot_ind_spec=False)



pca, scaler = initialize_pca()
data1 = ML_data(snr=0)
data2 = ML_data(snr=10)
data3 = ML_data(snr=0, evol_model = False)
data4 = ML_data(snr=10, evol_model = False)
data = [data1, data2, data3, data4]#, data5, data6]
model = define_pca_model(pca.n_components_, l1 = 1e-3, l2 = 1e-3)
model, pout = fit_model_plot_loop(model, data, n_loops = 1001, pca = pca, scaler = scaler)






#This works reasonably well
pca, scaler = initialize_pca()
data1 = ML_data(snr=0)
data2 = ML_data(snr=20)
data3 = ML_data(snr=0, evol_model = False)
data4 = ML_data(snr=20, evol_model = False)
data = [data1, data2, data3, data4]#, data5, data6]
model = define_pca_model(pca.n_components_, l1 = 1e-3, l2 = 1e-3)
model, pout = fit_model_plot_loop(model, data, n_loops = 1001, pca = pca, scaler = scaler)


#THIS SETUP WORKS REALLY WELL
n_chunks = 25
#data0 = ML_data(n_chunks = n_chunks, snr=0, generate_random = True, nrandom_templates = 5000)
data1 = ML_data(n_chunks = n_chunks, snr=0)
data2 = ML_data(n_chunks = n_chunks, snr=10)
#data3 = ML_data(n_chunks = n_chunks, snr=20)
data4 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False)
data5 = ML_data(n_chunks = n_chunks, snr=10, evol_model = False)
#data6 = ML_data(n_chunks = n_chunks, snr=20, evol_model = False)
#data = [data0, data1, data2, data3, data4, data5, data6]
data = [data1, data2, data4, data5]
model = define_split_model_plus(data[0], l1 = 1e-6, l2 = 1e-6)
model, pout = fit_model_plot_loop(model, data, n_loops = 201, plot_ind_spec = False)


#Only use constant SFR - Z models
n_chunks = 25
data1 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False)
data2 = ML_data(n_chunks = n_chunks, snr=20, evol_model = False)
data = [data1, data2]
model = define_split_model_plus(data[0], l1 = 1e-6, l2 = 1e-6)
model, pout = fit_model_plot_loop(model, data, n_loops = 201, plot_ind_spec = False)


#newest training data
n_chunks = 25
data1 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False, const_sfr = True)
data2 = ML_data(n_chunks = n_chunks, snr=20, evol_model = False, const_sfr = True)
data = [data1, data2]
model = define_split_model_plus(data[0], l1 = 1e-6, l2 = 1e-6)
model, pout = fit_model_plot_loop(model, data, n_loops = 2001, plot_ind_spec = False)




n_loops = 30
for i in range(n_loops):
    features, labels = combine_ml_data(data, pca = pca, scaler = scaler)
    mmm = model.fit(features, labels, batch_size = 128, epochs = 1,
                    validation_split=0.05, verbose=1)
    print(i)



n_chunks = 25 # set to 0 if using cnn
data0 = ML_data(n_chunks = n_chunks, snr=0)
data1 = ML_data(n_chunks = n_chunks, snr=50)
data2 = ML_data(n_chunks = n_chunks, snr=10)
data3 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False)
data4 = ML_data(n_chunks = n_chunks, snr=50, evol_model = False)
data5 = ML_data(n_chunks = n_chunks, snr=10, evol_model = False)
data = [data0, data1, data2, data3, data4, data5]
model = define_split_model_plus(data[0], l1 = 0, l2 = 0)
model = fit_model_plot_loop(model, data, n_loops = 25000)


n_chunks = 2 # set to 0 if using cnn
data0 = ML_data(n_chunks = n_chunks, snr=0, add_dust_extinction = True)
data1 = ML_data(n_chunks = n_chunks, snr=15, add_dust_extinction = True)
data2 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False, add_dust_extinction = True)
data3 = ML_data(n_chunks = n_chunks, snr=15, evol_model = False, add_dust_extinction = True)
data = [data0, data1, data2, data3]
model = define_split_model_plus_ebv(data[0], l1 = 0, l2 = 0)
model = fit_model_plot_loop(model, data, n_loops = 2500)



n_chunks = 1 # set to 0 if using cnn
data0 = ML_data(n_chunks = n_chunks, snr=0, add_dust_extinction = True)
data1 = ML_data(n_chunks = n_chunks, snr=15, add_dust_extinction = True)
data2 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False, add_dust_extinction = True)
data3 = ML_data(n_chunks = n_chunks, snr=15, evol_model = False, add_dust_extinction = True)
data = [data0, data1, data2, data3]
model = define_dense_model()
model = fit_model_plot_loop(model, data, n_loops = 2500)

#########To fit model in single pass
#BATCH_SIZE = 32
#EPOCHS = 200
#VALIDATION_SPLIT = 0.05
#model.fit(features, labels, batch_size = BATCH_SIZE, epochs = EPOCHS,
#          validation_split=VALIDATION_SPLIT, verbose=1)



###############Read in training and test data
#snr = 0 # no noise is added if SNR = 0
#n_chunks = 200 # set to 0 if using cnn
#data = ML_data(n_chunks = n_chunks, snr=snr)
#model = define_split_model(data, l1 = 0, l2 = 0)
#model = define_split_model(data, l1 = 1e-5, l2 = 1e-5)
#model = fit_model_plot_loop(model, data, n_loops = 50)


#########To fit model in for loop with plotting of test data


##########Return model prediction for individual spectra
spec, sdss = data[0].get_individual_sdss_spectra()
ppp = model.predict(spec)

############plot median MZ relation from individual spectra
sind = np.argsort(sdss['mass'])
ppp = ppp[sind,:]
sdss = sdss[sind]
zzz = ppp[:,0]
mmm = sdss['mass']
#good = np.where(zzz > -2 )
#zzz = zzz[good]
#mmm = mmm[good]
zsplit = np.array_split(zzz, 100)
msplit = np.array_split(mmm, 100)
zmed = []
mmed =[]
for i in np.arange(100):
    zmed.append(np.nanmedian(zsplit[i]))
    mmed.append(np.nanmedian(msplit[i]))
plot.plot(mmed,zmed)

########save model weights
#model_file  = '/Users/jabran/ml/metallicity/data/CNN_model.h5'
#model.save(model_file)


