%run ML_data.py
%run define_cnn_model.py
%run define_split_model.py
%run fit_model_plot_loop.py
#exec(open("/Users/jabran/ml/metallicity/programs/ml_model_fsps_spectra.py").read())

###############Read in training and test data
snr = 10 # no noise is added if SNR = 0
n_chunks = 100 # set to 0 if using cnn
n_filters = 1# set to 0 if using cnn
#this is instance of the data class
#data actually read in the fitting step
data = ML_data(n_chunks = n_chunks, n_filters = n_filters, snr=snr)

#data_demo = ML_data(n_chunks = 1, n_filters = 0, snr=0)
#features, labels = data_demo.get_training_data()
#spec, mass = data_demo.get_test_data()

#########Define which model to use
#model = define_cnn_model()
#model = define_split_model(data, l1 = 0, l2 = 0)
model = define_split_model(data, l1 = 5e-5, l2 = 5e-5)


#########To fit model in single pass
#BATCH_SIZE = 32
#EPOCHS = 200
#VALIDATION_SPLIT = 0.05
#model.fit(features, labels, batch_size = BATCH_SIZE, epochs = EPOCHS,
#          validation_split=VALIDATION_SPLIT, verbose=1)


#########To fit model in for loop with plotting of test data
model = fit_model_plot_loop(model, data, n_loops = 200)


##########Return model prediction for individual spectra
spec, sdss = data.get_individual_sdss_spectra()
ppp = model.predict(spec)

############plot median MZ relation from individual spectra
sind = np.argsort(sdss['mass'])
ppp = ppp[sind,:]
sdss = sdss[sind]
zzz = ppp[:,0]
mmm = sdss['mass']
good = np.where(zzz > -2 )
zzz = zzz[good]
mmm = mmm[good]
zsplit = np.array_split(zzz, 100)
msplit = np.array_split(mmm, 100)
zmed = []
mmed =[]
for i in np.arange(100):
    zmed.append(np.nanmedian(zsplit[i]))
    mmed.append(np.nanmedian(msplit[i]))


########save model weights
#model_file  = '/Users/jabran/ml/metallicity/data/CNN_model.h5'
#model.save(model_file)


