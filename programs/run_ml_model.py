#Compile module
%run /Users/jabran/ml/metallicity/programs/ml_model_fsps_spectra.py

#Read in training and test data
features, labels = get_training_data(norm = True)
test_data, mass = get_test_data(norm= True)

#Here is the training step
model = define_cnn_model()
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(features, labels, batch_size = BATCH_SIZE, epochs = EPOCHS,
              validation_split=VALIDATION_SPLIT, verbose=1)

#Save model
MODEL_FILE  = '/Users/jabran/ml/metallicity/data/CNN_model.h5'
model.save(MODEL_FILE )


# Return Z and Age for test data
ppp = model.predict(test_data)

#Load SSB fit results from Zahid+2017 for comparison
file ='/Users/jabran/ml/metallicity/data/MZR_SSB_fit_Zahid2017.txt'
mz = pd.read_csv(file)


#Plot predicted results along with 2017 SSB MZR
plot.plot(mass, np.log10(ppp[:,0]), label='ML')
plot.plot(mz['mass'], mz['Z'], label='SSB')
plot.ylabel('[Z/Z_solar]')
plot.xlabel('Stellar Mass')
plot.legend()
plot.show()


