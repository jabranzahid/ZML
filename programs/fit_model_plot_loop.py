from ML_data import ML_data
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


def fit_model_plot_loop(model, data, n_loops = 50):

    test_data, mass = data.get_test_data()
    features, labels = data.get_training_data()
    mz = data.get_zahid2017_mz()


    plot.ion()
    # set up the figure
    fig = plot.figure()
    plot.ylabel('[Z/Z_solar]')
    plot.xlabel('Stellar Mass')
    plot.show(block=False)

    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.01

    #This shows a movie of the convergence plotting the test set
    for i in range(n_loops):
        model.fit(features, labels, batch_size = BATCH_SIZE, epochs = 1,
              validation_split=VALIDATION_SPLIT, verbose=1)
        ppp = model.predict(test_data)
        plot.clf()
        plot.ion()
        plot.plot(mass, (ppp[:,0]), label='ML')
        plot.plot(mz['mass'], mz['Z'], label='SSB')
        plot.legend()
        plot.draw()
        mypause(0.01)
        print(i)

    return model
