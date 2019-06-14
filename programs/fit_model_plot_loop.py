from ML_data import ML_data
from ML_data import combine_training_data
from ML_data import combine_validation_data
from ML_data import transform_data_pca
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


def fit_model_plot_loop(model, data, n_loops = 50,
                        BATCH_SIZE = 128, plot_ind_spec = True,
                        training_data_fraction = 1, pca = None , scaler = None,
    ):

    fval, lval = combine_validation_data(data, pca = pca, scaler = scaler)
    val_data = (fval, lval)

    test_data, mass = data[0].get_test_data()
    test_data2, mass2 = data[0].get_test_data(shels_data = True)

    mz = data[0].get_zahid2017_mz()
    if plot_ind_spec:
        spec, sdss = data[0].get_individual_sdss_spectra()
        sind = np.argsort(sdss['mass'])

    if (pca != None) & (scaler != None):
        if plot_ind_spec:
            spec =np.squeeze(spec)
            inf_ind = np.where(spec == float('Inf'))
            spec[inf_ind] = 0
            spec = transform_data_pca(spec, pca, scaler)
        test_data = transform_data_pca(test_data[0], pca, scaler)
        test_data2 = transform_data_pca(test_data2[0], pca, scaler)


    plot.ion()
    # set up the figure
    fig = plot.figure()
    plot.show(block=False)

    pout = []
    mout = []

    frac = [0.010190666, 0.052520990, 0.12342441, 0.29004723, 0.52381670]
    frac_arr = (np.tile(frac, 25)).reshape(25, 5)
    #This shows a movie of the convergence plotting the test set
    for i in range(n_loops):

        features, labels = combine_training_data(data, pca = pca, scaler = scaler)

        mmm = model.fit(features, labels, batch_size = BATCH_SIZE, epochs = 1,
                         validation_data = val_data, verbose=1)
        #mout.append(mmm)
        ppp = model.predict(test_data)
        pout.append(ppp)
        ppp2 = model.predict(test_data2)
        plot.clf()
        #plot.ion()
        #plot.plot(mass, np.sum(ppp[:,0:5]*frac_arr, axis=1), label='ML MW SDSS')
        #plot.plot(mass, (ppp[:,4]), label='ML SDSS OLD')
        plot.plot(mass, (ppp[:,0]), label='ML SDSS')
        #plot.plot(mass2, (ppp2[:,0]), label='ML SHELS')
        plot.plot(mz['mass'], mz['Z'], label='SSB SDSS')
        plot.ylim(-0.8, 0.3)
        plot.ylabel('[Z/Z_solar]')
        plot.xlabel('Stellar Mass')
        plot.legend()
        plot.draw()

        if plot_ind_spec:
            if i%25 == 0:
                ppp = model.predict(spec)
                ppp = ppp[sind,:]
                zzz = ppp[:,0]
                mmm = sdss['mass']
                mmm = mmm[sind]
                #good = np.where(zzz > -2 )
                #zzz = zzz[good]
                #mmm = mmm[good]
                zsplit = np.array_split(zzz, 100)
                msplit = np.array_split(mmm, 100)
                zmed = []
                mmed =[]
                for j in np.arange(100):
                    zmed.append(np.nanmedian(zsplit[j]))
                    mmed.append(np.nanmedian(msplit[j]))

            plot.plot(mmed,zmed)

        mypause(0.00001)
        print(i)

    return model, pout#, mout
