#Compile module
%run /Users/jabran/ml/metallicity/programs/ml_model_fsps_spectra.py

#Read in training and test data
SNR = 0
features, labels = get_training_data(norm = True, SNR = SNR)
test_data, mass = get_test_data(norm= True)

#Here is the training step
model = define_cnn_model()
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.05
model.fit(features, labels, batch_size = BATCH_SIZE, epochs = EPOCHS,
          validation_split=VALIDATION_SPLIT, verbose=1)

#Save model
MODEL_FILE  = '/Users/jabran/ml/metallicity/data/CNN_model.h5'
model.save(MODEL_FILE )


# Return Z and Age for test data
ppp = model.predict(test_data)

#Load SSB fit results from Zahid+2017 for comparison
file ='/Users/jabran/ml/metallicity/data/MZR_SSB_fit_Zahid2017.txt'

#Load SSB fit results but with 0.1 Myr burst duration
file ='/Users/jabran/ml/metallicity/data/MZR_SSB_fit_burst0.1.txt'

mz = pd.read_csv(file)


#Plot predicted results along with 2017 SSB MZR
plot.plot(mass, np.log10(ppp[:,0]), label='ML')
plot.plot(mz['mass'], mz['Z'], label='SSB')
plot.ylabel('[Z/Z_solar]')
plot.xlabel('Stellar Mass')
plot.legend()
plot.show()



#This shows a movie of the convergence plotting the test set
for i in range(100):
    model.fit(features, labels, batch_size = BATCH_SIZE, epochs = 1,
          validation_split=VALIDATION_SPLIT, verbose=1)
    ppp = model.predict(test_data)
    plot.clf()
    plot.ion()
    plot.plot(mass, np.log10(ppp[:,0]), label='ML')
    plot.plot(mz['mass'], mz['Z'], label='SSB')
    plot.ylabel('[Z/Z_solar]')
    plot.xlabel('Stellar Mass')
    plot.draw()
    plot.pause(1)
    print(i)
