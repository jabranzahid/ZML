#Compile module
#%run /Users/jabran/ml/metallicity/programs/ml_model_fsps_spectra.py
exec(open("/Users/jabran/ml/metallicity/programs/ml_model_fsps_spectra.py").read())
#Read in training and test data
SNR = 0
features, labels = get_training_data(norm = True, SNR = SNR)
test_data, mass = get_test_data(norm= True)

#Load SSB fit results from Zahid+2017 for comparison
file ='/Users/jabran/ml/metallicity/data/MZR_SSB_fit_Zahid2017.txt'
#Load SSB fit results but with 0.1 Myr burst duration
#file ='/Users/jabran/ml/metallicity/data/MZR_SSB_fit_burst0.1.txt'
mz = pd.read_csv(file)


#Here is the training step
model = define_cnn_model()
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.05


plot.ion()
# set up the figure
fig = plot.figure()
plot.ylabel('[Z/Z_solar]')
plot.xlabel('Stellar Mass')
plot.show(block=False)


import matplotlib
matplotlib.use("Qt5agg") # or "Qt5agg" depending on you version of Qt
def mypause(interval):
    manager = plot._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        #plt.show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


#This shows a movie of the convergence plotting the test set
for i in range(100):
    model.fit(features, labels, batch_size = BATCH_SIZE, epochs = 1,
          validation_split=VALIDATION_SPLIT, verbose=1)
    ppp = model.predict(test_data)
    plot.clf()
    plot.ion()
    plot.plot(mass, np.log10(ppp[:,0]), label='ML')
    plot.plot(mz['mass'], mz['Z'], label='SSB')
    plot.draw()
    mypause(1)
    print(i)
