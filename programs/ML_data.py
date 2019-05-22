from astropy.io import fits
import numpy as np
import pandas as pd


DATA_FILE_PATH = '/Users/jabran/ml/metallicity/data/'

class ML_data:

    def __init__(self, n_chunks = 1, n_filters = 1, snr = 0, norm = True):
        self.data_file_path = DATA_FILE_PATH
        self.n_chunks = n_chunks
        self.n_filters = n_filters
        self.snr = snr
        self.norm = norm
        #this came from mask_emission_sdss_andrews.pro
        mask_file = "/Users/jabran/ml/metallicity/data/emission_line_mask.txt"
        mask = np.loadtxt(mask_file)
        self.mask_index = np.where(mask == 0)


    def split_data_into_chunks(self, input_arr):

        output_arr=[]

        output_arr1 = np.squeeze(input_arr)
        output_arr1 = np.array_split(output_arr1, self.n_chunks, axis=1)
        for i in range(self.n_filters): output_arr.extend(output_arr1)
        return output_arr

    def data_shape(self):
        shape1=[]
        shape=[]
        eg_arr = np.array_split(np.empty(2482), self.n_chunks)
        for x in eg_arr: shape1.append(x.shape)
        for i in range(self.n_filters): shape.extend(shape1)

        return shape



    def get_training_data(self, SNR = 0, zmax = 0.5):
    #read in training data
    #these files were produced in IDL using wrapper_fsps_sfh_z_tabular_ascii.pro
    # and make_evol_fsps_model_str.pro. I fiddled with the programs,
    #one set is normalized the other is not using /no_norm keyword

        if self.norm == False:
            FILE = "fsps_evol_models_no_norm.fits"
        else:
            FILE = "fsps_evol_models_norm.fits"

        FSPS_FILE = self.data_file_path + FILE
        hdul = fits.open(FSPS_FILE)
        data = hdul[1].data
        flux = data.field(1)

        #if SNR set, then add gaussian noise with a set SNR
        if self.snr != 0:
            noise = np.random.normal(size=(1030, 3249))*(flux/self.snr)
        else:
            noise = 0

        flux = flux + noise - 1
        z_solar =0.0142
        lwz = np.log10(data['LWZ']/z_solar)
        lwa = np.log10(data['LWA'])

        index = self.mask_index

        flux_mask = (flux[:,index].reshape(1030, 2482))
        index_zlo = np.where(lwz < zmax)
        nsel = len(index_zlo[0])
        flux_mask = (flux_mask[index_zlo,:].reshape(nsel, 2482))
        features = np.expand_dims(flux_mask, axis=2)
        n_flux = len(flux_mask[0,:])
        n_spec = len(flux_mask)
        labels = np.stack((lwz[index_zlo], lwa[index_zlo]), axis=1)

        #randomly reshuffle before feeding CNN
        np.random.seed(4)
        np.random.shuffle(features)
        np.random.seed(4)
        np.random.shuffle(labels)

        if self.n_chunks > 0: features = self.split_data_into_chunks(features)

        return features, labels




    def get_test_data(self, high_mass = True, sfr_sort = False):


        if sfr_sort:
            if self.norm:
                FILE = "sdss_sort_stack_data_norm.fits"
            else:
                FILE = "sdss_sort_stack_data_no_norm.fits"
        else:
            if self.norm:
                FILE = "sdss_stack_data_norm.fits"
            else:
                FILE = "sdss_stack_data_no_norm.fits"


        STACK_FILE = self.data_file_path + FILE

        hdul = fits.open(STACK_FILE)
        data = hdul[1].data
        flux_test = data.field(1) - 1
        mass = data['MASS']

        index = self.mask_index

        if sfr_sort:
            flux_test_mask = (flux_test[:,index].reshape(170, 2482))
            flux_test_mask = np.expand_dims(flux_test_mask, axis=2)
        else:
            flux_test_mask = (flux_test[:,index].reshape(34, 2482))
            flux_test_mask = np.expand_dims(flux_test_mask, axis=2)


        if high_mass:
            if sfr_sort:
                ind = 45
            else:
                ind = 9
            flux_test_mask = flux_test_mask[ind:,:,:]
            mass = mass[ind:]

        if self.n_chunks > 0 : flux_test_mask = self.split_data_into_chunks(flux_test_mask)


        return flux_test_mask, mass





    def get_individual_sdss_spectra(self):

        FILE1 = "sdss_individual_spectra.fits"
        FILE2 = "sdss_individual_spectra_str.fits"

        SPEC_FILE = self.data_file_path + FILE1
        SDSS_FILE = self.data_file_path + FILE2

        hdul = fits.open(SPEC_FILE, memmap=True)
        flux_spec = hdul[0].data - 1
        n_spec = len(flux_spec)

        hdul2 = fits.open(SDSS_FILE)
        sdss = hdul2[1].data

        index = self.mask_index

        flux_spec = (flux_spec[:,index].reshape(n_spec, 2482))
        flux_spec = np.expand_dims(flux_spec, axis=2)

        if self.n_chunks > 0 : flux_spec = self.split_data_into_chunks(flux_spec)

        return flux_spec, sdss



    def get_zahid2017_mz(self, burst = False):

        #Load SSB fit results from Zahid+2017 for comparison
        file = 'MZR_SSB_fit_Zahid2017.txt'
        if burst: file ='MZR_SSB_fit_burst0.1.txt'

        mzfile = self.data_file_path + file
        mz = pd.read_csv(mzfile)

        return mz




