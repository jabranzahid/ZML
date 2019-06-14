from astropy.io import fits
from dust_extinction.parameter_averages import O94
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyval, polyfit
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA as kPCA
from sklearn.preprocessing import StandardScaler as SCALER


DATA_FILE_PATH = '/Users/jabran/ml/metallicity/data/'

def initialize_pca():


    #data0 = ML_data(generate_random = True, nrandom_templates = 3000)
    #data1 = ML_data()
    #data2 = ML_data(evol_model = False)
    #data = [data0, data1, data2]

    #fpca, lpca = combine_ml_data(data)
    #f = fpca[0] + 1

    f, z, age = ML_data().generate_single_burst_data()
    f += 1
    scaler = SCALER()
    scaler.fit(f)
    f = scaler.transform(f)

    pca = PCA(n_components=8)
    pca.fit(f)


#!!!!!This sequence did not significantly improve results
#    evol, labels = ML_data().get_training_data()
#    evol = evol[0] + 1
#    evol_scaler = scaler.transform(evol)
#    evol_pca = pca.transform(evol_scaler)
#    evol_inverse = pca.inverse_transform(evol_pca)
#    diff = evol_scaler - evol_inverse

#    fff = np.append(f, diff, axis=0)
#    pca.fit(fff)


    return pca, scaler


def transform_data_pca(features, pca, scaler):

    ff = features + 1
    ff = scaler.transform(ff)
    ff = pca.transform(ff)
    features = [ff]

    return features

def combine_training_data(obj_list, pca = None, scaler = None):
    nobj = len(obj_list)
    if nobj == 1:
        f0, l0 = obj_list[0].get_training_data()
        features = []
        labels = np.row_stack((l0))
        for i in range(len(f0)): features.append(np.row_stack((f0[i])))
    if nobj == 2:
        f0, l0 = obj_list[0].get_training_data()
        f1, l1 = obj_list[1].get_training_data()
        features = []
        labels = np.row_stack((l0,l1))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i])))
    if nobj == 3:
        f0, l0 = obj_list[0].get_training_data()
        f1, l1 = obj_list[1].get_training_data()
        f2, l2 = obj_list[2].get_training_data()
        features = []
        labels = np.row_stack((l0,l1, l2))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i])))
    if nobj == 4:
        f0, l0 = obj_list[0].get_training_data()
        f1, l1 = obj_list[1].get_training_data()
        f2, l2 = obj_list[2].get_training_data()
        f3, l3 = obj_list[3].get_training_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i])))
    if nobj == 5:
        f0, l0 = obj_list[0].get_training_data()
        f1, l1 = obj_list[1].get_training_data()
        f2, l2 = obj_list[2].get_training_data()
        f3, l3 = obj_list[3].get_training_data()
        f4, l4 = obj_list[4].get_training_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3, l4))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i], f4[i])))
    if nobj == 6:
        f0, l0 = obj_list[0].get_training_data()
        f1, l1 = obj_list[1].get_training_data()
        f2, l2 = obj_list[2].get_training_data()
        f3, l3 = obj_list[3].get_training_data()
        f4, l4 = obj_list[4].get_training_data()
        f5, l5 = obj_list[5].get_training_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3, l4, l5))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i], f4[i], f5[i])))
    if nobj == 7:
        f0, l0 = obj_list[0].get_training_data()
        f1, l1 = obj_list[1].get_training_data()
        f2, l2 = obj_list[2].get_training_data()
        f3, l3 = obj_list[3].get_training_data()
        f4, l4 = obj_list[4].get_training_data()
        f5, l5 = obj_list[5].get_training_data()
        f6, l6 = obj_list[6].get_training_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3, l4, l5, l6))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i], f4[i], f5[i], f6[i])))

    index = np.arange(len(features[0][:,0]))
    np.random.shuffle(index)
    labels = labels[index, :]
    for i in range(len(features)): features[i] = features[i][index,:]

    if (pca != None) & (scaler != None):
        ff = features[0]
        features = transform_data_pca(ff, pca, scaler)

    return features, labels




def combine_validation_data(obj_list, pca = None, scaler = None):
    nobj = len(obj_list)
    if nobj == 1:
        f0, l0 = obj_list[0].get_validation_data()
        features = []
        labels = np.row_stack((l0))
        for i in range(len(f0)): features.append(np.row_stack((f0[i])))
    if nobj == 2:
        f0, l0 = obj_list[0].get_validation_data()
        f1, l1 = obj_list[1].get_validation_data()
        features = []
        labels = np.row_stack((l0,l1))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i])))
    if nobj == 3:
        f0, l0 = obj_list[0].get_validation_data()
        f1, l1 = obj_list[1].get_validation_data()
        f2, l2 = obj_list[2].get_validation_data()
        features = []
        labels = np.row_stack((l0,l1, l2))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i])))
    if nobj == 4:
        f0, l0 = obj_list[0].get_validation_data()
        f1, l1 = obj_list[1].get_validation_data()
        f2, l2 = obj_list[2].get_validation_data()
        f3, l3 = obj_list[3].get_validation_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i])))
    if nobj == 5:
        f0, l0 = obj_list[0].get_validation_data()
        f1, l1 = obj_list[1].get_validation_data()
        f2, l2 = obj_list[2].get_validation_data()
        f3, l3 = obj_list[3].get_validation_data()
        f4, l4 = obj_list[4].get_validation_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3, l4))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i], f4[i])))
    if nobj == 6:
        f0, l0 = obj_list[0].get_validation_data()
        f1, l1 = obj_list[1].get_validation_data()
        f2, l2 = obj_list[2].get_validation_data()
        f3, l3 = obj_list[3].get_validation_data()
        f4, l4 = obj_list[4].get_validation_data()
        f5, l5 = obj_list[5].get_validation_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3, l4, l5))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i], f4[i], f5[i])))
    if nobj == 7:
        f0, l0 = obj_list[0].get_validation_data()
        f1, l1 = obj_list[1].get_validation_data()
        f2, l2 = obj_list[2].get_validation_data()
        f3, l3 = obj_list[3].get_validation_data()
        f4, l4 = obj_list[4].get_validation_data()
        f5, l5 = obj_list[5].get_validation_data()
        f6, l6 = obj_list[6].get_validation_data()
        features = []
        labels = np.row_stack((l0, l1, l2, l3, l4, l5, l6))
        for i in range(len(f0)): features.append(np.row_stack((f0[i], f1[i], f2[i], f3[i], f4[i], f5[i], f6[i])))

    index = np.arange(len(features[0][:,0]))
    np.random.shuffle(index)
    labels = labels[index, :]
    for i in range(len(features)): features[i] = features[i][index,:]

    if (pca != None) & (scaler != None):
        ff = features[0]
        features = transform_data_pca(ff, pca, scaler)

    return features, labels




def add_noise_to_training_data(arr_in, snr):
    np.random.seed(np.random.randint(100000))
    noise = np.random.normal(size=np.shape(arr_in))/snr
    arr_out = arr_in + noise
    return arr_out


def split_data_into_chunks(input_arr, n_chunks, n_filters):

    output_arr=[]

    output_arr1 = np.squeeze(input_arr, axis=2)
    output_arr1 = np.array_split(output_arr1, n_chunks, axis=1)
    for i in range(n_filters): output_arr.extend(output_arr1)
    return output_arr


def get_instance_training_data(ML_obj):

    labels = ML_obj.raw_labels
    #if SNR set, then add gaussian noise with a set SNR
    if ML_obj.snr != 0:
        flux_mask = add_noise_to_training_data(ML_obj.raw_features, ML_obj.snr)
    else:
        flux_mask = ML_obj.raw_features

    features = np.expand_dims(flux_mask, axis=2)

    if ML_obj.training_data_fraction != 1:
        ind_max = round(ML_obj.n_spec*ML_obj.training_data_fraction)
        features = features[0:ind_max-1,:,:]
        labels = labels[0:ind_max-1,:]

    if ML_obj.n_chunks > 0 or ML_obj.n_filters > 1:
        features = split_data_into_chunks(features, ML_obj.n_chunks, ML_obj.n_filters)

    return features, np.asarray(labels)




class ML_data:

    def __init__(
        self, n_chunks = 1, snr = 0, mask_lines = True,
        training_data_fraction = 1, evol_model = True,
        add_dust_extinction = False, generate_random = False,
        nrandom_templates = 20000, validation_split = 0.1, n_random_avg = 5,
    ):

        self.z_solar = 0.0142 # appropriate solar metallicity for these models
        self.data_file_path = DATA_FILE_PATH
        self.training_data_fraction = 1 #this is defunct, clean up later
        self.n_chunks = n_chunks
        self.n_filters = 1 #this keyword is defunct, add neurons to first layer
        self.snr = snr
        self.add_dust_extinction = add_dust_extinction
        self.nrandom_templates = nrandom_templates
        self.validation_split = validation_split
        self.n_random_avg = n_random_avg
        #this came from mask_emission_sdss_andrews.pro
        mask_file = DATA_FILE_PATH + "emission_line_mask.txt"
        mask = np.loadtxt(mask_file)
        if mask_lines:
            self.mask_index = (np.where(mask == 0))[0]
        else:
            self.mask_index = (np.where(mask >= 0))[0]
        self.n_flux = len(self.mask_index)
        if generate_random:
            self.n_labels = 10
            self.generate_random_training_data()
        else:
            self.read_training_data(evol_model = evol_model)



    def get_training_data(self):
        features, labels = get_instance_training_data(self)
        return features, labels



    def get_validation_data(self):
        labels_val = self.raw_labels_val
        if self.snr != 0:
            flux_mask_val = add_noise_to_training_data(self.raw_features_val, self.snr)
        else:
            flux_mask_val = self.raw_features_val

        features_val = np.expand_dims(flux_mask_val, axis=2)

        if self.n_chunks > 0 or self.n_filters > 1:
            features_val = split_data_into_chunks(features_val, self.n_chunks, self.n_filters)

        return features_val, labels_val




    def split_training_validation(self, flux, labels):

        val_split = 1 - self.validation_split
        n_spec = len(flux)
        ind_max = round(n_spec*val_split)

        flux_train = flux[0:ind_max, :]
        labels_train = labels[0:ind_max, :]
        flux_val = flux[ind_max:, :]
        labels_val = labels[ind_max:, :]

        return flux_train, labels_train, flux_val, labels_val




    def data_shape(self):
        shape1=[]
        shape=[]
        eg_arr = np.array_split(np.empty(self.n_flux), self.n_chunks)
        for x in eg_arr: shape1.append(x.shape)
        for i in range(self.n_filters): shape.extend(shape1)

        return shape


    def extinguish_spectra(self, flux, labels, ebv_range = [0,0.4], nebv = 5, Rv=3.1):

        wave_inv_micron = 1./(self.wave*1e-4)
        ebv = np.linspace(ebv_range[0], ebv_range[1], nebv)
        flux_out = np.zeros(shape=(self.n_spec*nebv, self.n_flux))
        labels_out = np.zeros(shape=(self.n_spec*nebv, self.n_labels+1))

        for i in range(nebv):
            spec_ext = O94(Rv=Rv).extinguish(wave_inv_micron, Ebv=ebv[i])
            spec_ext_mat = np.outer( np.ones((self.n_spec,)), spec_ext )
            flux_out[i*self.n_spec:(i+1)*self.n_spec,:] = spec_ext_mat*flux
            labels_out[i*self.n_spec:(i+1)*self.n_spec, 0:self.n_labels] = labels
            labels_out[i*self.n_spec:(i+1)*self.n_spec,self.n_labels] = ebv[i]

        ind_norm = (np.where((self.wave >= 4400) & (self.wave <= 4450)))[0]
        med_norm = np.median(flux_out[:,ind_norm], axis=1)
        flux_norm = np.outer( med_norm, np.ones((self.n_flux,)))
        flux_out /= flux_norm

        return flux_out, labels_out


    def generate_single_burst_data(self, zmax = 0.41):

        FILE = "fsps_burst_templates_01.0.fits"
        FSPS_FILE = self.data_file_path + FILE

        hdul = fits.open(FSPS_FILE)
        data = hdul[1].data
        templates = np.squeeze(data['flux'], axis=0)
        age = np.squeeze(data['age'])
        wave = np.squeeze(data['wave'])
        z = np.squeeze(data['z'])/self.z_solar
        lum = np.squeeze(data['lum'])
        pfit_ind = np.squeeze(data['pfit_index'])


        z_ind = (np.where(  (np.log10(z) < zmax)))[0]
        age_ind = np.arange(len(age))
        z_ind = np.arange(len(z))
        age = age[age_ind]
        z = z[z_ind]
        nz = len(z)
        nage = len(age)

        templates = templates[:,age_ind,:]
        templates = templates[z_ind,:,:]
        lum = lum[:,age_ind]
        lum = lum[z_ind,:]


        temp_norm = (np.moveaxis(templates, 2,0)/lum).reshape(3349, nz*nage)

        ndeg = 20
        wave_fit = np.arange(len(wave), dtype='float64')/(len(wave)-1) - 0.5
        coeffs = polyfit(wave_fit[pfit_ind] , temp_norm[pfit_ind,:], ndeg)
        pfit = (polyval(wave_fit,coeffs)).T

        flux = (temp_norm/pfit)[50:-50]
        flux = (flux[self.mask_index,:]).T - 1

        return flux, age, z




    def generate_random_training_data(self, zmax = 0.41):

        FILE = "fsps_burst_templates_01.0.fits"
        FSPS_FILE = self.data_file_path + FILE
        ntemplates = self.nrandom_templates
        n_avg = self.n_random_avg

        hdul = fits.open(FSPS_FILE)
        data = hdul[1].data
        templates = np.moveaxis(np.squeeze(data['flux'], axis=0), 2, 0)
        age = np.squeeze(data['age'])
        wave = np.squeeze(data['wave'])
        z = np.squeeze(data['z'])/self.z_solar
        pfit_ind = np.squeeze(data['pfit_index'])
        temp_lum = np.squeeze(data['lum'])

        dt = np.repeat(age*(10**0.05), len(z)).reshape(len(age), len(z)).T
        norm_temp = templates*dt
        temp_lum *= dt

        #age_ind = np.concatenate((np.arange(3)*8+33, np.arange(53)+51))
        #age_ind = np.arange(50)+51
        age_ind = np.concatenate((np.arange(5)*5+5,np.arange(72)+30))
        z_ind = (np.where(  (np.log10(z) < zmax) & (np.log10(z) > -1.1)))[0]
        age = age[age_ind]
        z = z[z_ind]
        norm_temp = norm_temp[:,:,age_ind]
        norm_temp = norm_temp[:,z_ind,:]
        temp_lum = temp_lum[:,age_ind]
        temp_lum = temp_lum[z_ind,:]

        nz = len(z)
        nage = len(age)
        n_labels2 = int(self.n_labels/2)
        nrandom = ntemplates*n_labels2*n_avg

        #ind_z = np.flip(np.sort((np.random.randint(0, nz, nrandom)).reshape(ntemplates, n_labels2*n_avg), axis=1))
        #ind_age = np.sort((np.random.randint(0, nage, nrandom)).reshape(ntemplates, n_labels2*n_avg), axis=1)
        ind_z = []
        ind_age = []
        for i in range(ntemplates):
            nz_min = np.random.randint(0, nz/3)
            nz_max = np.random.randint(2*nz/3+1, nz)
            ind_z1 = np.flip(np.sort(np.random.randint(nz_min, nz_max +1, n_labels2*n_avg)))
            ind_z1 = np.asarray(np.split(ind_z1, n_labels2))
            np.random.shuffle(ind_z1)
            ind_z1 = ind_z1.reshape(n_labels2*n_avg)
            ind_z.append(ind_z1)
            nage_min = np.random.randint(0, nage/2)
            nage_max = np.random.randint(2*nage/3+1, nage)
            ind_age.append(np.sort(np.random.randint(nage_min, nage_max+1, n_labels2*n_avg)))

        ind_z = np.asarray(ind_z)
        ind_age = np.asarray(ind_age)

        la_rand = np.log10(age[ind_age])
        lz_rand = np.log10(z[ind_z])
        rand_temp = norm_temp[:, ind_z, ind_age]
        rand_lum = temp_lum[ind_z, ind_age]
        if n_avg > 1:
            rand_lum2 = np.random.uniform(size = (ntemplates,n_labels2*n_avg))*0 + 1
            rand_temp = np.sum(np.asarray(np.split(rand_temp*rand_lum2, n_labels2, axis=2)), axis=3)
            rand_temp = np.moveaxis(rand_temp.T, 1,0)
            lz_rand = np.sum(np.asarray(np.split(lz_rand*rand_lum*rand_lum2, n_labels2, axis=1)), axis=2)/np.sum(np.asarray(np.split(rand_lum*rand_lum2, n_labels2, axis=1)), axis=2)
            lz = lz_rand.T
            la_rand = np.sum(np.asarray(np.split(la_rand*rand_lum*rand_lum2, n_labels2, axis=1)), axis=2)/np.sum(np.asarray(np.split(rand_lum*rand_lum2, n_labels2, axis=1)), axis=2)
            la = la_rand.T
        else:
            la = la_rand
            lz = lz_rand
        labels = np.column_stack((lz,la))

        wave_ind = (np.where((wave >= 4400) & (wave <= 4450)))[0]
        med_norm = np.median(rand_temp[wave_ind,:,:], axis=0)
        rand_temp /= med_norm
        norm_temp = np.sum(rand_temp, axis=2)/5

        ndeg = 20
        wave_fit = np.arange(len(wave), dtype='float64')/(len(wave)-1) - 0.5
        coeffs = polyfit(wave_fit[pfit_ind] , norm_temp[pfit_ind,:], ndeg)
        pfit = (polyval(wave_fit,coeffs)).T

        flux = (norm_temp/pfit)[50:-50]
        flux = (flux[self.mask_index,:]).T

        if self.add_dust_extinction:
            flux, labels = self.extinguish_spectra(flux, labels)

        if self.validation_split != 0:
            flux_train, labels_train, flux_val, labels_val = self.split_training_validation(flux, labels)
            flux_train -= 1
            flux_val -= 1
        else:
            flux_train = flux - 1
            labels_train = labels
            flux_val = [None]
            labels_val = [None]

        self.raw_features = tuple(flux_train)
        self.raw_labels = tuple(labels_train)
        self.raw_features_val = tuple(flux_val)
        self.raw_labels_val = tuple(labels_val)




    def read_training_data(self, zmax = 0.41, only_z_zero = False, evol_model = True, label_plus = True):
        #read in training data
        #these files were produced in IDL using wrapper_fsps_sfh_z_tabular_ascii.pro
        #and make_evol_fsps_model_str.pro. I fiddled with the programs,
        #one set is normalized the other is not using /no_norm keyword

        if self.add_dust_extinction:
            if evol_model:
                FILE = "fsps_evol_models_no_norm.fits"
            else:
                FILE = 'fsps_constant_z_sfr_models_no_norm.fits'
        else:
            if evol_model:
                FILE = "fsps_evol_models_norm.fits"
            else:
                FILE = 'fsps_constant_z_sfr_models_norm2.fits'


        FSPS_FILE = self.data_file_path + FILE
        hdul = fits.open(FSPS_FILE)
        data = hdul[1].data

        if only_z_zero and evol_model:
            ind_z_zero = np.where(data['time_ind'] == 7)
            data = data[ind_z_zero]

        flux = data.field(1)
        wave = data['wave'][0,:]

        lwa = data['LWA1']
        mwa  = data['LOGMWA']
        lwz = data['LWZ1']/self.z_solar
        mwz  = data['LOGMWZ']/self.z_solar

        if label_plus:
            labels_lin = np.column_stack((mwz,mwa))
        else:
            labels_lin = np.stack((lwz, lwa), axis=1)

        labels = np.log10(labels_lin )

        index_zlo = (np.where(np.log10(lwz) < zmax))[0]
        flux = flux[index_zlo, :]
        flux = flux[:, self.mask_index]
        labels = labels[index_zlo,:]

        self.wave = wave[self.mask_index]
        self.n_spec = len(flux[:,0])
        self.n_labels = len(labels[0,:])

        if self.add_dust_extinction:
            flux, labels = self.extinguish_spectra(flux, labels)

        #randomly reshuffle due to how validation data is selected
        np.random.seed(4)
        np.random.shuffle(flux)
        np.random.seed(4)
        np.random.shuffle(labels)

        if self.validation_split != 0:
            flux_train, labels_train, flux_val, labels_val = self.split_training_validation(flux, labels)
            flux_train -= 1
            flux_val -= 1
        else:
            flux_train = flux - 1
            labels_train = labels
            flux_val = [None]
            labels_val = [None]

        self.raw_features = tuple(flux_train)
        self.raw_labels = tuple(labels_train)
        self.raw_features_val = tuple(flux_val)
        self.raw_labels_val = tuple(labels_val)



    def get_test_data(self, high_mass = True, sfr_sort = False, shels_data = False):


        if self.add_dust_extinction:
            if sfr_sort:
                FILE = "sdss_sort_stack_data_no_norm.fits"
            else:
                FILE = "sdss_stack_data_no_norm.fits"
        else:
            if sfr_sort:
                FILE = "sdss_sort_stack_data_norm.fits"
            else:
                FILE = "sdss_stack_data_norm.fits"

        if shels_data:
            FILE = "shels_stack_data_norm.fits"

        STACK_FILE = self.data_file_path + FILE

        hdul = fits.open(STACK_FILE)
        data = hdul[1].data
        flux_test = data.field(1) - 1
        mass = data['MASS']

        index = self.mask_index

        if sfr_sort:
            flux_test_mask = np.expand_dims(flux_test[:,index], axis=2)
        else:
            flux_test_mask = np.expand_dims(flux_test[:,index], axis=2)

        if not shels_data:
            if high_mass:
                if sfr_sort:
                    ind = 45
                else:
                    ind = 9
                flux_test_mask = flux_test_mask[ind:,:,:]
                mass = mass[ind:]

        if self.n_chunks > 0 or self.n_filters > 1:
            flux_test_mask = split_data_into_chunks(flux_test_mask, self.n_chunks, self.n_filters)


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

        flux_spec = (flux_spec[:,index].reshape(n_spec, self.n_flux))
        flux_spec = np.expand_dims(flux_spec, axis=2)

        if self.n_chunks > 1 or self.n_filters > 1:
            flux_spec = split_data_into_chunks(flux_spec, self.n_chunks, self.n_filters)

        return flux_spec, sdss



    def get_zahid2017_mz(self, burst = False):

        #Load SSB fit results from Zahid+2017 for comparison
        file = 'MZR_SSB_fit_Zahid2017.txt'
        if burst: file ='MZR_SSB_fit_burst0.1.txt'

        mzfile = self.data_file_path + file
        mz = pd.read_csv(mzfile)

        return mz
