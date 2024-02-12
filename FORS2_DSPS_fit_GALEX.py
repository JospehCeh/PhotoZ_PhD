#!/usr/bin/env python
# coding: utf-8

# # Fit Fors2 Spectra and Photometry with DSPS
# Restricted to FORS2 galaxies with GALEX photometry

# Implement this fit using this `fors2tostellarpopsynthesis` package
# 
# - Author Joseph Chevalier
# - Afflilation : IJCLab/IN2P3/CNRS
# - Organisation : LSST-DESC
# - creation date : 2024-01-10
# - last update : 2024-01-10 : Initial version
# 
# Most functions are inside the package. This notebook inherits largely from `Fors2ToStellarPopSynthesis/docs/notebooks/fitters/FitFors2ManySpecLoop.ipynb` in the `fors2tostellarpopsynthesis` package.

# ## Imports and general settings
import h5py
import pandas as pd
import numpy as np
import os
import re
import pickle
import copy
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import collections
from collections import OrderedDict
import matplotlib.gridspec as gridspec
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import vmap
import jaxopt
import optax
jax.config.update("jax_enable_x64", True)
from interpax import interp1d

plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['xtick.labelsize']= 'xx-large'
plt.rcParams['ytick.labelsize']= 'xx-large'
plt.rcParams['legend.fontsize']=  16

kernel = kernels.RBF(0.5, (8000, 20000.0))
gpr = GaussianProcessRegressor(kernel=kernel ,random_state=0)


# ## Filters
from fors2tostellarpopsynthesis.filters import FilterInfo

from fors2tostellarpopsynthesis.fors2starlightio import Fors2DataAcess,\
                                                        SLDataAcess,\
                                                        convert_flux_torestframe,\
                                                        gpr


# ## Fitter with jaxopt
from fors2tostellarpopsynthesis.fitters.fitter_jaxopt import (lik_spec_ageDepMet_Q,\
                                                              lik_spec_from_mag_ageDepMet_Q,\
                                                              lik_normspec_from_mag_ageDepMet_Q,\
                                                              lik_mag_ageDepMet_Q,\
                                                              lik_ugri_ageDepMet_Q,\
                                                              lik_comb_ageDepMet_Q,\
                                                              get_infos_spec,\
                                                              get_infos_mag,\
                                                              get_infos_comb)

from fors2tostellarpopsynthesis.fitters.fitter_jaxopt import (SSP_DATA,\
                                                              mean_spectrum_ageDepMet_Q,\
                                                              mean_mags_ageDepMet_Q,\
                                                              mean_ugri_ageDepMet_Q,\
                                                              mean_sfr_ageDepMet_Q,\
                                                              ssp_spectrum_fromparam_ageDepMet_Q)

from fors2tostellarpopsynthesis.fitters.fitter_util import (plot_fit_ssp_photometry_ageDepMet_Q,\
                                                            plot_fit_ssp_ugri_ageDepMet_Q,\
                                                            plot_fit_ssp_spectrophotometry_ageDepMet_Q,\
                                                            plot_fit_ssp_spectrophotometry_sl_ageDepMet_Q,\
                                                            plot_fit_ssp_spectroscopy_ageDepMet_Q,\
                                                            plot_SFH_ageDepMet_Q,\
                                                            rescale_photometry_ageDepMet_Q,\
                                                            rescale_spectroscopy_ageDepMet_Q,\
                                                            rescale_starlight_inrangefors2)


# ## Parameters to fit
from fors2tostellarpopsynthesis.parameters import (SSPParametersFit_AgeDepMet_Q,\
                                                   paramslist_to_dict)
        
def plot_figs_to_PDF(pdf_file, fig_list):
    with PdfPages(pdf_file) as pdf:
        for fig in fig_list :
            pdf.savefig(fig)
            plt.close()

FLAG_REMOVE_GALEX = False
FLAG_REMOVE_GALEX_FUV = True
FLAG_REMOVE_VISIBLE = False

def main(args):
    if len(args) < 3:
        low_bound, high_bound = 1, 5
    else :
        low_bound, high_bound = int(args[1]), int(args[2])
    ps = FilterInfo()
    ps.plot_transmissions()

    # ## FORS2 and Starlight SPS extrapolation
    # ### Observed FORS2 data
    fors2 = Fors2DataAcess()
    fors2_tags = fors2.get_list_of_groupkeys()
    list_of_fors2_attributes = fors2.get_list_subgroup_keys()

    # ### Extrapolated Starlight data
    sl = SLDataAcess()
    sl_tags = sl.get_list_of_groupkeys()

    # ## Select applicable spectra
    filtered_tags = []
    for idx, tag in enumerate(fors2_tags):
        fors2_attr = fors2.getattribdata_fromgroup(tag)

        bool_viz = FLAG_REMOVE_VISIBLE or (not(FLAG_REMOVE_VISIBLE)\
                                           and np.isfinite(fors2_attr['MAG_GAAP_u'])\
                                           and np.isfinite(fors2_attr['MAG_GAAP_g'])\
                                           and np.isfinite(fors2_attr['MAG_GAAP_r'])\
                                           and np.isfinite(fors2_attr['MAG_GAAP_i'])\
                                           and np.isfinite(fors2_attr['MAGERR_GAAP_u'])\
                                           and np.isfinite(fors2_attr['MAGERR_GAAP_g'])\
                                           and np.isfinite(fors2_attr['MAGERR_GAAP_r'])\
                                           and np.isfinite(fors2_attr['MAGERR_GAAP_i']))

        bool_fuv = (FLAG_REMOVE_GALEX or FLAG_REMOVE_GALEX_FUV) or (not(FLAG_REMOVE_GALEX or FLAG_REMOVE_GALEX_FUV)\
                                                                    and np.isfinite(fors2_attr['fuv_mag'])\
                                                                    and np.isfinite(fors2_attr['fuv_magerr']))

        bool_nuv = FLAG_REMOVE_GALEX or (not(FLAG_REMOVE_GALEX)\
                                         and np.isfinite(fors2_attr['nuv_mag'])\
                                         and np.isfinite(fors2_attr['nuv_magerr']))

        if bool_viz and bool_fuv and bool_nuv :
            filtered_tags.append(tag)
    #random_tags = np.random.choice(selected_tags, size=5, replace=False)

    #usr_sel = input(f"There are {len(filtered_tags)} appropriate spectra to fit. Please specify the interval you wish to fit as follows :\nx y\nwhere all spectra from the x-th to the y-th will be fitted, starting from 1.\n")

    #low_bound, high_bound = tuple(int(n) for n in usr_sel.split(" "))
    print(len(filtered_tags))
    low_bound -= 1
    high_bound = min(high_bound, len(filtered_tags))
    low_bound = min(low_bound, high_bound-1)

    selected_tags = filtered_tags[low_bound:high_bound]
    start_tag, end_tag = selected_tags[0], selected_tags[-1]

# ## Attempt with fewer parameters and age-dependant, fixed-bounds metallicity
    dict_fors2_for_fit = {}
    for tag in tqdm(selected_tags):
        dict_tag = {}
        # extract most basic info
        selected_spectrum_number = int(re.findall("^SPEC(.*)", tag)[0])
        fors2_attr = fors2.getattribdata_fromgroup(tag)
        z_obs = fors2_attr['redshift']
        title_spec = f"{tag} z = {z_obs:.2f}"

        dict_tag["spec ID"] = selected_spectrum_number
        dict_tag["redshift"] = z_obs
        dict_tag["title"] = title_spec

        # retrieve magnitude data
        data_mags, data_magserr = fors2.get_photmagnitudes(tag)
        ugri_mags_c, ugri_magserr_c = fors2.get_ugrimagnitudes_corrected(tag) # KiDS ugri magnitudes
                                                                            # corrected for dust extinction

        # get the Fors2 spectrum
        spec_obs = fors2.getspectrumcleanedemissionlines_fromgroup(tag)
        Xs = spec_obs['wl']
        Ys = spec_obs['fnu']
        EYs = spec_obs['bg']
        EYs_med = spec_obs['bg_med']

        # convert to restframe
        Xspec_data, Yspec_data = convert_flux_torestframe(Xs, Ys, z_obs)
        EYspec_data = EYs #* (1+z_obs)
        EYspec_data_med = EYs_med #* (1+z_obs)

        dict_tag["wavelengths"] = Xspec_data
        dict_tag["fnu"] = Yspec_data
        dict_tag["fnu_err"] = EYspec_data

        # smooth the error over the spectrum
        #fit_res = gpr.fit(Xspec_data[:, None], EYspec_data)
        #EYspec_data_sm = gpr.predict(Xspec_data[:, None], return_std=False)

        # need to increase error to decrease chi2 error
        #EYspec_data_sm *= 2

        # Choose filters with mags without Nan
        NoNaN_mags = np.intersect1d(np.argwhere(~np.isnan(data_mags)).flatten(),\
                                    np.argwhere(~np.isnan(data_magserr)).flatten())

        # selected indexes for filters
        index_selected_filters = NoNaN_mags

        if FLAG_REMOVE_GALEX:
            galex_indexes = np.array([0,1])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)
        elif FLAG_REMOVE_GALEX_FUV:
            galex_indexes = np.array([0])
            index_selected_filters = np.setdiff1d(NoNaN_mags, galex_indexes)    

        if FLAG_REMOVE_VISIBLE:
            visible_indexes = np.array([2, 3, 4, 5, 6, 7])
            index_selected_filters = np.setdiff1d(NoNaN_mags, visible_indexes)

        # Select filters
        XF = ps.get_2lists()
        NF = len(XF[0])
        list_wls_f_sel = []
        list_trans_f_sel = []
        list_name_f_sel = []
        list_wlmean_f_sel = []

        for index in index_selected_filters:
            list_wls_f_sel.append(XF[0][index])
            list_trans_f_sel.append(XF[1][index])
            the_filt = ps.filters_transmissionlist[index]
            the_wlmean = the_filt.wave_mean
            list_wlmean_f_sel.append(the_wlmean)
            list_name_f_sel.append(ps.filters_namelist[index])

        list_wlmean_f_sel = jnp.array(list_wlmean_f_sel)
        Xf_sel = (list_wls_f_sel, list_trans_f_sel)

        NoNan_ugri = np.intersect1d(NoNaN_mags, np.array([2, 3, 4, 5]))
        list_wls_ugri = []
        list_trans_ugri = []
        list_name_ugri = []
        list_wlmean_ugri = []

        for index in NoNan_ugri:
            list_wls_ugri.append(XF[0][index])
            list_trans_ugri.append(XF[1][index])
            the_filt = ps.filters_transmissionlist[index]
            the_wlmean = the_filt.wave_mean
            list_wlmean_ugri.append(the_wlmean)
            list_name_ugri.append(ps.filters_namelist[index])

        list_wlmean_ugri = jnp.array(list_wlmean_ugri)
        Xf_ugri = (list_wls_ugri, list_trans_ugri)
        print(NoNan_ugri, list_name_ugri)

        # get the magnitudes and magnitude errors
        data_selected_mags =  jnp.array(data_mags[index_selected_filters])
        data_selected_magserr = jnp.array(data_magserr[index_selected_filters])
        data_selected_ugri_corr =  jnp.array(ugri_mags_c)
        data_selected_ugri_correrr = jnp.array(ugri_magserr_c)

        dict_tag["filters"] = Xf_sel
        dict_tag["wl_mean_filters"] = list_wlmean_f_sel
        dict_tag["mags"] = data_selected_mags
        dict_tag["mags_err"] = data_selected_magserr
        dict_tag["ugri_filters"] = Xf_ugri
        dict_tag["wl_mean_ugri"] = list_wlmean_ugri
        dict_tag["ugri_corr"] = data_selected_ugri_corr
        dict_tag["ugri_corr_err"] = data_selected_ugri_correrr

        dict_fors2_for_fit[tag] = dict_tag

    # parameters for fit
    pdfoutputfilename = f"DSPS_pickles/fitparams_galex_successive_fits_{low_bound+1}-{start_tag}_to_{high_bound}-{end_tag}.pdf"
    list_of_figs = []
    p = SSPParametersFit_AgeDepMet_Q()
    init_params = p.INIT_PARAMS
    params_min = p.PARAMS_MIN
    params_max = p.PARAMS_MAX
    lbfgsb_ugri = jaxopt.ScipyBoundedMinimize(fun = lik_ugri_ageDepMet_Q, method = "L-BFGS-B", maxiter=5000)
    lbfgsb_mag = jaxopt.ScipyBoundedMinimize(fun = lik_mag_ageDepMet_Q, method = "L-BFGS-B", maxiter=5000)
    lbfgsb_spec = jaxopt.ScipyBoundedMinimize(fun = lik_normspec_from_mag_ageDepMet_Q, method = "L-BFGS-B", maxiter=5000)
    lbfgsb_comb = jaxopt.ScipyBoundedMinimize(fun = lik_comb_ageDepMet_Q, method = "L-BFGS-B", maxiter=5000)

    # fit loop
    for tag in tqdm(dict_fors2_for_fit):
        data_dict = dict_fors2_for_fit[tag]

        # fit with magnitudes only
        res_m = lbfgsb_ugri.run(init_params,\
                                bounds=(params_min, params_max),\
                                xf = data_dict["ugri_filters"],\
                                mags_measured = data_dict["ugri_corr"],\
                                sigma_mag_obs = data_dict["ugri_corr_err"],\
                                z_obs = data_dict["redshift"])
        '''
        params_m, fun_min_m, jacob_min_m, inv_hessian_min_m = get_infos_mag(res_m,\
                                                                            lik_mag_ageDepMet_Q,\
                                                                            xf = data_dict["filters"],\
                                                                            mgs = data_dict["mags"],\
                                                                            mgse = data_dict["mags_err"],\
                                                                            z_obs=data_dict["redshift"])
        '''
        #print("params:",params_m,"\nfun@min:",fun_min_m,"\njacob@min:",jacob_min_m)

        # Convert fitted parameters into a dictionnary
        params_m = res_m.params
        dict_params_m = paramslist_to_dict(params_m, p.PARAM_NAMES_FLAT)

        '''
        # rescale photometry datapoints
        xphot_rest, yphot_rest, eyphot_rest, factor\
        = rescale_photometry_ageDepMet_Q(dict_params_m,\
                                         dict_tag["wl_mean_filters"],\
                                         data_dict["mags"],\
                                         data_dict["mags_err"],\
                                         data_dict["redshift"])

        #rescale Fors2 spectroscopy
        Xspec_data_rest, Yspec_data_rest, EYspec_data_rest, factor\
        = rescale_spectroscopy_ageDepMet_Q(dict_params_m,\
                                           data_dict["wavelengths"],\
                                           data_dict["fnu"],\
                                           data_dict["fnu_err"],\
                                           data_dict["redshift"])
        '''

        '''
        # plot SFR
        plot_SFH_ageDepMet_Q(dict_params_m, data_dict["redshift"], subtit = data_dict["title"], ax=None)

        # plot fit for photometry only
        plot_fit_ssp_photometry_ageDepMet_Q(dict_params_m,\
                                            data_dict["filters"],\
                                            data_dict["wl_mean_filters"],\
                                            data_dict["mags"],\
                                            data_dict["mags_err"],\
                                            data_dict["redshift"],\
                                            data_dict["title"])
        '''
        # fit spectroscopy

        ''' Joseph
        Remplacement des *spec_data_rest ci-dessous par *spec_data suite à réintroduction du paramètre SCALE
        '''
        p_to_fit = jnp.array([_par for _par in params_m[-3:]])
        p_fixed = jnp.array([_par for _par in params_m[:-3]])
        res_s = lbfgsb_spec.run(p_to_fit,\
                                bounds = (params_min[-3:], params_max[-3:]),\
                                p_fix = p_fixed,\
                                wls = data_dict["wavelengths"],\
                                F = data_dict["fnu"],\
                                sigma_obs = data_dict["fnu_err"],\
                                z_obs = data_dict["redshift"])
        '''
        params_s, fun_min_s, jacob_min_s, inv_hessian_min_s = get_infos_spec(res_s,\
                                                                             lik_spec_ageDepMet_Q,\
                                                                             wls = data_dict["wavelengths"],\
                                                                             F = data_dict["fnu"],\
                                                                             eF = data_dict["fnu_err"],\
                                                                             z_obs = data_dict["redshift"])
        '''
        #print("params:",params_s,"\nfun@min:",fun_min_s,"\njacob@min:",jacob_min_s)

        # Convert fitted parameters with spectroscopy into a dictionnary
        params_s = res_s.params
        dict_params_s = paramslist_to_dict(jnp.concatenate((params_m[:-3], params_s)), p.PARAM_NAMES_FLAT)

        p_to_fit = jnp.array([_par for _par in params_m[:-3]])
        p_fixed = jnp.array([_par for _par in params_s])
        res_mm = lbfgsb_mag.run(p_to_fit,\
                                bounds = (params_min[:-3], params_max[:-3]),\
                                p_fix = p_fixed,\
                                xf = data_dict["filters"],\
                                mags_measured = data_dict["mags"],\
                                sigma_mag_obs = data_dict["mags_err"],\
                                z_obs = data_dict["redshift"])
        params_mm = res_mm.params
        dict_params_mm = paramslist_to_dict(jnp.concatenate((params_mm, params_s)), p.PARAM_NAMES_FLAT)

        # rescale photometry datapoints
        '''
        xphot_rest, yphot_rest, eyphot_rest, factor\
        = rescale_photometry_ageDepMet_Q(dict_params_s,\
                                         data_dict["wl_mean_filters"],\
                                         data_dict["mags"],\
                                         data_dict["mags_err"],\
                                         data_dict["redshift"])
        '''
        xphot_rest = data_dict["wl_mean_filters"]
        yphot_rest, eyphot_rest = data_dict["mags"], data_dict["mags_err"]

        #rescale Fors2 spectroscopy
        Xspec_data_rest, Yspec_data_rest, EYspec_data_rest, factor\
        = rescale_spectroscopy_ageDepMet_Q(dict_params_mm,\
                                           data_dict["wavelengths"],\
                                           data_dict["fnu"],\
                                           data_dict["fnu_err"],\
                                           data_dict["redshift"])

        # plot SFR
        f, axs = plt.subplots(2, 1, figsize=(12, 12))
        axs = axs.ravel()
        plot_SFH_ageDepMet_Q(dict_params_mm, data_dict["redshift"], subtit = data_dict["title"], ax=axs[0])

        #load starlight spectrum
        dict_sl = sl.getspectrum_fromgroup(tag)

        # rescale starlight spectrum
        w_sl, fnu_sl, _ = rescale_starlight_inrangefors2(dict_sl["wl"],\
                                                         dict_sl["fnu"],\
                                                         Xspec_data_rest,\
                                                         Yspec_data_rest)

        # plot all final data + starlight
        plot_fit_ssp_spectrophotometry_sl_ageDepMet_Q(dict_params_mm,\
                                                      Xspec_data_rest,\
                                                      Yspec_data_rest,\
                                                      EYspec_data_rest,\
                                                      data_dict["filters"],\
                                                      xphot_rest,\
                                                      yphot_rest,\
                                                      eyphot_rest,\
                                                      w_sl,\
                                                      fnu_sl,\
                                                      data_dict["redshift"],\
                                                      data_dict["title"],\
                                                      ax=axs[1])


        #save to dictionary
        dict_out = OrderedDict()
        dict_out["fors2name"] = tag
        dict_out["zobs"] = data_dict["redshift"]
        Ns = len(Yspec_data_rest)
        dict_out["Ns"] = Ns
        #dict_out["funcmin_s"] = fun_min_s

        # convert into a dictionnary
        #dict_out.update(dict_params_s)
        #dict_out.update(dict_params_m)
        dict_out.update(dict_params_mm)

        '''
        # combining spectro and photometry
        Xc = [Xspec_data_rest, Xf_sel]
        Yc = [Yspec_data_rest, data_selected_mags]
        EYc = [EYspec_data_rest, data_selected_magserr]
        weight_spec = 0.5
        Ns = len(Yspec_data_rest)
        Nm = len(data_selected_mags)
        Nc = Ns+Nm

        # do the combined fit
        lbfgsb = jaxopt.ScipyBoundedMinimize(fun = lik_comb_ageDepMet_Q, method = "L-BFGS-B")
        res_c = lbfgsb.run(init_params,\
                           bounds = (params_min, params_max),\
                           xc = Xc,\
                           datac = Yc,\
                           sigmac = EYc,\
                           z_obs = z_obs,\
                           weight = weight_spec)
        params_c, fun_min_c, jacob_min_c, inv_hessian_min_c = get_infos_comb(res_c,\
                                                                             lik_comb_ageDepMet_Q,\
                                                                             xc = Xc,\
                                                                             datac = Yc,\
                                                                             sigmac = EYc,\
                                                                             z_obs = z_obs,\
                                                                             weight = weight_spec)
        params_cm, fun_min_cm, jacob_min_cm, inv_hessian_min_cm  = get_infos_mag(res_c,\
                                                                                 lik_mag_ageDepMet_Q,\
                                                                                 xf = Xf_sel,\
                                                                                 mgs = data_selected_mags,\
                                                                                 mgse = data_selected_magserr,\
                                                                                 z_obs = z_obs)
        params_cs, fun_min_cs, jacob_min_cs, inv_hessian_min_cs = get_infos_spec(res_c,\
                                                                                 lik_spec_ageDepMet_Q,\
                                                                                 wls = Xspec_data_rest,\
                                                                                 F = Yspec_data_rest,\
                                                                                 eF = EYspec_data_rest,\
                                                                                 z_obs = z_obs)
        print("params_c:", params_c, "\nfun@min:", fun_min_c, "\njacob@min:", jacob_min_c) #,"\n invH@min:",inv_hessian_min_c)
        print("params_cm:", params_cm, "\nfun@min:", fun_min_cm, "\njacob@min:", jacob_min_cm)
        print("params_cs:", params_cs, "\nfun@min:", fun_min_cs, "\njacob@min:", jacob_min_cs)

        #save to dictionary
        dict_out = OrderedDict()
        dict_out["fors2name"] = tag
        dict_out["zobs"] = z_obs
        dict_out["Nc"] = Nc
        dict_out["Ns"] = Ns
        dict_out["Nm"] = Nm
        dict_out["funcmin_c"] = fun_min_c
        dict_out["funcmin_m"] = fun_min_cm
        dict_out["funcmin_s"] = fun_min_cs

        # convert into a dictionnary
        dict_params_c = paramslist_to_dict(params_c, p.PARAM_NAMES_FLAT) 
        dict_out.update(dict_params_c)

        # plot the combined fit
        plot_fit_ssp_spectrophotometry_ageDepMet_Q(dict_params_c,\
        Xspec_data_rest,\
                                       Yspec_data_rest,\
                                       EYspec_data_rest,\
                                                    data_dict["filters"],\
                                       xphot_rest,\
                                       yphot_rest,\
                                       eyphot_rest,\
                                       z_obs = z_obs,\
                                       subtit = title_spec )
        '''
        #save figures and parameters
        list_of_figs.append(copy.deepcopy(f))

        filename_params = f"DSPS_pickles/fitparams_{tag}_galex_ageDepMet_Q_successive_fits.pickle"
        with open(filename_params, 'wb') as f:
            #print(dict_out)
            pickle.dump(dict_out, f)
    plot_figs_to_PDF(pdfoutputfilename, list_of_figs)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
    