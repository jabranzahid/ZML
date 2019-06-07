from ML_data import ML_data
from ML_data import combine_ml_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
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


def fit_model_plot_loop(model, data, n_loops = 50, BATCH_SIZE = 128, VALIDATION_SPLIT = 0.05, training_data_fraction = 1):

    test_data, mass = data[0].get_test_data()
    test_data2, mass2 = data[0].get_test_data(shels_data = True)

    mz = data[0].get_zahid2017_mz()
    spec, sdss = data[0].get_individual_sdss_spectra()
    sind = np.argsort(sdss['mass'])



    plot.ion()
    # set up the figure
    fig = plot.figure()
    plot.show(block=False)

    pout = []

    frac = [0.010190666, 0.052520990, 0.12342441, 0.29004723, 0.52381670]
    frac_arr = (np.tile(frac, 25)).reshape(25, 5)
    #This shows a movie of the convergence plotting the test set
    for i in range(n_loops):

        features, labels = combine_ml_data(data)
        if len(features) == 1: features = features[0]
        model.fit(features, labels, batch_size = BATCH_SIZE, epochs = 1,
                   validation_split=VALIDATION_SPLIT, verbose=1)
        ppp = model.predict(test_data)
        pout.append(ppp)
        ppp2 = model.predict(test_data2)
        plot.clf()
        plot.ion()
#        plot.plot(mass, np.sum(ppp[:,0:5]*frac_arr, axis=1), label='ML MW SDSS')
#        plot.plot(mass, (ppp[:,4]), label='ML SDSS')
        plot.plot(mass, (ppp[:,0]), label='ML SDSS')
        #plot.plot(mass2, (ppp2[:,0]), label='ML SHELS')
        plot.plot(mz['mass'], mz['Z'], label='SSB SDSS')
        plot.ylabel('[Z/Z_solar]')
        plot.xlabel('Stellar Mass')
        plot.legend()
        plot.draw()

 #       if i%1e5 == 0:
 #           ppp = model.predict(spec)
 #           ppp = ppp[sind,:]
 #           zzz = ppp[:,0]
 #           mmm = sdss['mass']
 #           mmm = mmm[sind]
            #good = np.where(zzz > -2 )
            #zzz = zzz[good]
            #mmm = mmm[good]
#            zsplit = np.array_split(zzz, 100)
#            msplit = np.array_split(mmm, 100)
#            zmed = []
#            mmed =[]
#            for i in np.arange(100):
#                zmed.append(np.nanmedian(zsplit[i]))
#                mmed.append(np.nanmedian(msplit[i]))

        #plot.plot(mmed,zmed)

        mypause(0.01)
        print(i)

    return model, pout
