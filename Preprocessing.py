from __future__ import print_function
import numpy as np
import spectral as spy
 #
class Processor:
    def __init__(self):
        pass

    def save_res_4kfolds_cv(self, y_pres, y_tests, file_name=None, verbose=False):
        """
        save experiment results for k-folds cross validation
        :param y_pres: predicted labels, k*Ntest
        :param y_tests: true labels, k*Ntest
        :param file_name:
        :return:
        """
        ca, oa, aa, kappa = [], [], [], []
        for y_p, y_t in zip(y_pres, y_tests):
            ca_, oa_, aa_, kappa_ = self.score(y_t, y_p)
            ca.append(np.asarray(ca_)), oa.append(np.asarray(oa_)), aa.append(np.asarray(aa_)),
            kappa.append(np.asarray(kappa_))
        ca = np.asarray(ca) * 100
        oa = np.asarray(oa) * 100
        aa = np.asarray(aa) * 100
        kappa = np.asarray(kappa)
        ca_mean, ca_std = np.round(ca.mean(axis=0), 2), np.round(ca.std(axis=0), 2)
        oa_mean, oa_std = np.round(oa.mean(), 2), np.round(oa.std(), 2)
        aa_mean, aa_std = np.round(aa.mean(), 2), np.round(aa.std(), 2)
        kappa_mean, kappa_std = np.round(kappa.mean(), 3), np.round(kappa.std(), 3)
        if file_name is not None:
            file_name = 'scores.npz'
            np.savez(file_name, y_test=y_tests, y_pre=y_pres,
                     ca_mean=ca_mean, ca_std=ca_std,
                     oa_mean=oa_mean, oa_std=oa_std,
                     aa_mean=aa_mean, aa_std=aa_std,
                     kappa_mean=kappa_mean, kappa_std=kappa_std)
            print('the experiments have been saved in ', file_name)

        if verbose is True:
            print('---------------------------------------------')
            print('ca\t\t', '\taa\t\t', '\toa\t\t', '\tkappa\t\t')
            print(ca_mean, '+-', ca_std)
            print(aa_mean, '+-', aa_std)
            print(oa_mean, '+-', oa_std)
            print(kappa_mean, '+-', kappa_std)

        # return ca, oa, aa, kappa
        return np.asarray([ca_mean, ca_std]), np.asarray([aa_mean, aa_std]), \
               np.asarray([oa_mean, oa_std]), np.asarray([kappa_mean, kappa_std])