import time, sys
import os
import numpy as np
import pandas as pd
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from collections import OrderedDict
import re
import h5py
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# ---------------
ordered_keys = ['name','num','ra','dec', 'redshift','Rmag','RT', 'RV','eRV','Nsp','lines',
                'ra_galex','dec_galex','fuv_mag', 'fuv_magerr','nuv_mag', 'nuv_magerr', 
                'fuv_flux', 'fuv_fluxerr','nuv_flux', 'nuv_fluxerr','asep_galex',
                'ID', 'KIDS_TILE','RAJ2000','DECJ2000','Z_ML', 'Z_B','asep_kids','CLASS_STAR',
                'MAG_GAAP_u','MAG_GAAP_g','MAG_GAAP_r','MAG_GAAP_i','MAG_GAAP_Z','MAG_GAAP_Y','MAG_GAAP_J', 'MAG_GAAP_H','MAG_GAAP_Ks',
                'MAGERR_GAAP_u','MAGERR_GAAP_g','MAGERR_GAAP_r','MAGERR_GAAP_i','MAGERR_GAAP_Z','MAGERR_GAAP_Y','MAGERR_GAAP_J','MAGERR_GAAP_H','MAGERR_GAAP_Ks',
                'FLUX_GAAP_u','FLUX_GAAP_g','FLUX_GAAP_r','FLUX_GAAP_i','FLUX_GAAP_Z','FLUX_GAAP_Y','FLUX_GAAP_J', 'FLUX_GAAP_H','FLUX_GAAP_Ks',
                'FLUXERR_GAAP_u','FLUXERR_GAAP_g','FLUXERR_GAAP_r','FLUXERR_GAAP_i','FLUXERR_GAAP_Z','FLUXERR_GAAP_Y','FLUXERR_GAAP_J','FLUXERR_GAAP_H','FLUXERR_GAAP_Ks',
                'FLUX_RADIUS', 'EXTINCTION_u','EXTINCTION_g','EXTINCTION_r', 'EXTINCTION_i',]

#---------------
class Fors2DataAcess(object):
    def __init__(self,filename):
        if os.path.isfile(filename):
            self.hf = h5py.File(filename, 'r')
            self.list_of_groupkeys = list(self.hf.keys())      
             # pick one key    
            key_sel =  self.list_of_groupkeys[0]
            # pick one group
            group = self.hf.get(key_sel)  
            #pickup all attribute names
            self.list_of_subgroup_keys = []
            for k in group.attrs.keys():
                self.list_of_subgroup_keys.append(k)
        else:
            self.hf = None
            self.list_of_groupkeys = []
            self.list_of_subgroup_keys = []
    def close_file(self):
        self.hf.close() 
        
    def get_list_of_groupkeys(self):
        return self.list_of_groupkeys 
    def get_list_subgroup_keys(self):
        return self.list_of_subgroup_keys
    def getattribdata_fromgroup(self,groupname):
        attr_dict = OrderedDict()
        if groupname in self.list_of_groupkeys:       
            group = self.hf.get(groupname)  
            for  nameval in self.list_of_subgroup_keys:
                attr_dict[nameval] = group.attrs[nameval]
        else:
            print(f'getattribdata_fromgroup : No group {groupname}')
        return attr_dict
    def getspectrum_fromgroup(self,groupname):
        spec_dict = {}
        if groupname in self.list_of_groupkeys:       
            group = self.hf.get(groupname)  
            wl = np.array(group.get("wl"))
            fl = np.array(group.get("fl")) 
            spec_dict["wl"] = wl
            spec_dict["fl"] = fl
        else:
            print(f'getspectrum_fromgroup : No group {groupname}')
        return spec_dict
    
    #kernel = kernels.RBF(0.5, (8000, 10000.0))
    #gp = GaussianProcessRegressor(kernel=kernel ,random_state=0)

    def getspectrumcleanedemissionlines_fromgroup(self,groupname,gp,nsigs=8):
        spec_dict = {}
        if groupname in self.list_of_groupkeys:       
            group = self.hf.get(groupname)  
            wl = np.array(group.get("wl"))
            fl = np.array(group.get("fl")) 
            # fit gaussian pricess 
            X = wl
            Y = fl
            
            DeltaY = Y - gp.predict(X[:, None], return_std=False)
            background = np.sqrt(np.median(DeltaY**2))
            indexes_toremove = np.where(np.abs(DeltaY)> nsigs * background)[0]
            
            Xclean = np.delete(X,indexes_toremove)
            Yclean  = np.delete(Y,indexes_toremove)
            
            spec_dict["wl"] = Xclean
            spec_dict["fl"] = Yclean

            #remove negatives
            bad_indexes = np.where(Yclean<=0)[0]
            spec_dict["wl"] = np.delete(Xclean,  bad_indexes)
            spec_dict["fl"] = np.delete(Yclean,  bad_indexes)
            
        else:
            print(f'getspectrum_fromgroup : No group {groupname}')
        return spec_dict
        
        
           
         

# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose': True,
              'debug': False,
              'outfile': 'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee fitting parameters
              'nwalkers': 128,
              'nburn': [16, 32, 64],
              'niter': 512,
              'interval': 0.25,
              'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi',  # bounding method
              'nested_sample': 'unif',  # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_target_n_effective': 10000,
              # Obs data parameters
              'objid': 0,
              'phottable': 'demo_photometry.dat',
              'luminosity_distance': 1e-5,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# Model Definition
# --------------

def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
                add_neb=False, luminosity_distance=0.0, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # Add lumdist parameter.  If this is not added then the distance is
    # controlled by the "zred" parameter and a WMAP9 cosmology.
    if luminosity_distance > 0:
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units":"Mpc"}

    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 13.
    model_params["mass"]["init"] = 1e8

    # If we are going to be using emcee, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["mass"]["init_disp"] = 1e7
    model_params["tau"]["init_disp"] = 3.0
    model_params["tage"]["init_disp"] = 5.0
    model_params["tage"]["disp_floor"] = 2.0
    model_params["dust2"]["disp_floor"] = 0.1

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model

# --------------
# Observational Data
# --------------

# Here we are going to put together some filter names


galex = ['galex_FUV', 'galex_NUV']
vista = ['vista_vircam_'+n for n in ['Z','J','H','Ks']]
sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i']]
filternames = galex + sdss + vista

# The first filter set is Johnson/Cousins, the second is SDSS. We will use a
# flag in the photometry table to tell us which set to use for each object
# (some were not in the SDSS footprint, and therefore have Johnson/Cousins
# photometry)
#
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
filtersets = (galex +  sdss + vista)


def build_obs(objid=0, phottable='demo_photometry.dat',
              luminosity_distance=None, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)

    from prospect.utils.obsutils import fix_obs

    # Here we will read in an ascii catalog of magnitudes as a numpy structured
    # array
    with open(phottable, 'r') as f:
        # drop the comment hash
        header = f.readline().split()[1:]
    catalog = np.genfromtxt(phottable, comments='#',
                            dtype=np.dtype([(n, float) for n in header]))

    # Find the right row
    ind = catalog['objid'] == float(objid)
    # Here we are dynamically choosing which filters to use based on the object
    # and a flag in the catalog.  Feel free to make this logic more (or less)
    # complicated.
    filternames = filtersets[int(catalog[ind]['filterset'])]
    # And here we loop over the magnitude columns
    mags = [catalog[ind]['mag{}'.format(i)] for i in range(len(filternames))]
    mags = np.array(mags)
    # And since these are absolute mags, we can shift to any distance.
    if luminosity_distance is not None:
        dm = 25 + 5 * np.log10(luminosity_distance)
        mags += dm

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.squeeze(10**(-mags/2.5))
    # HACK.  You should use real flux uncertainties
    obs['maggies_unc'] = obs['maggies'] * 0.07
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None

    # Add unessential bonus info.  This will be stored in output
    #obs['dmod'] = catalog[ind]['dmod']
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))

input_file_h5  = '../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5'

if __name__ == '__main__':


    print("START MAIN")

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--inputfile', type=str, default="FORS2spectraGalexKidsPhotom.hdf5",
                        help="file containing photometry and spectroscopy for Fors2 objects.")
    parser.add_argument('--objid', type=int, default=0,
                        help="Object identifier number.")
    parser.add_argument('--datamode', type=str, default="photom",
                        help="type of data to fit : mode = photom/spectrophotom/spectro")

    args = parser.parse_args()
    print("args = ",args)

    object_number = int(args.objid)

    #retrieve the object number
    selected_key = f'SPEC{object_number}'

    input_file_h5 = args.inputfile

    if not os.path.isfile(input_file_h5):
        print(f"File not found {input_file_h5}")
        sys.exit(-1)

    if args.datamode not in ['photom','spectrophotom','spectro']:
        print(f"Bad datamode {args.datamode}")
        sys.exit(-1)

     #decode
    fors2 = Fors2DataAcess(input_file_h5)
    list_of_keys = fors2.get_list_of_groupkeys()
    list_of_attributes = fors2.get_list_subgroup_keys()
    list_of_keys = np.array(list_of_keys)
    list_of_keysnum = [ int(re.findall("SPEC(.*)",specname)[0]) for specname in  list_of_keys ]
    sorted_indexes = np.argsort(list_of_keysnum)
    list_of_keys = list_of_keys[sorted_indexes]    

    if selected_key in list_of_keys:
        print(f"Object {selected_key} found in file {input_file_h5}")


    kernel = kernels.RBF(0.5, (8000, 10000.0))
    gp = GaussianProcessRegressor(kernel=kernel ,random_state=0)

    attrs = fors2.getattribdata_fromgroup(selected_key)
    spectr = fors2.getspectrumcleanedemissionlines_fromgroup(selected_key,gp,nsigs=3.)
    df_info = pd.DataFrame(columns=list_of_attributes)
    df_info.loc[0] = [*attrs.values()] # hope the order of attributes is kept

    print(df_info)
    
    run_params = vars(args)

    print(run_params)
    
    sys.exit()
    
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

   
    #hfile = setup_h5(model=model, obs=obs, **run_params)
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "{0}_{1}_result.h5".format(args.outfile, ts)


    print(f"Outputfile {hfile}")
    sys.exit()

    output = fit_model(obs, model, sps, noise, **run_params)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
