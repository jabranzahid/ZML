%run ML_data.py
%run define_cnn_model.py
%run define_dense_model.py
%run define_split_model.py
%run define_split_model_plus.py
%run define_split_model_plus_ebv.py
%run fit_model_plot_loop.py

#THIS SETUP WORKS REALLY WELL
n_chunks = 25
data0 = ML_data(n_chunks = n_chunks, snr=0, generate_random = True, nrandom_templates = 5000)
data2 = ML_data(n_chunks = n_chunks, snr=0)
data3 = ML_data(n_chunks = n_chunks, snr=20)
data4 = ML_data(n_chunks = n_chunks, snr=0, evol_model = False)
data5 = ML_data(n_chunks = n_chunks, snr=20, evol_model = False)
data = [data0, data2, data3, data4, data5]
model = define_split_model_plus(data[0], l1 = 1e-6, l2 = 1e-6)
model, pout = fit_model_plot_loop(model, data, n_loops = 400)



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


