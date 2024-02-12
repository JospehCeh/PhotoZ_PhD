#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


from dsps import load_ssp_templates
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_sed_sfh_table_met_table
from dsps import load_transmission_curve
from dsps import calc_rest_mag
from dsps import calc_obs_mag
from dsps.dust.att_curves import (RV_C00,\
                                  _frac_transmission_from_k_lambda,\
                                  sbl18_k_lambda)

import jax
import jax.numpy as jnp
from jax import vmap, jit
import jaxopt
import optax
jax.config.update("jax_enable_x64", True)
from interpax import interp1d

from fors2tostellarpopsynthesis.filters import FilterInfo

from fors2tostellarpopsynthesis.fitters.fitter_jaxopt import (SSP_DATA,\
                                                              TODAY_GYR,\
                                                              mean_sfr,\
                                                              ssp_spectrum_fromparam,\
                                                              mean_sfr_ageDepMet_Q,\
                                                              ssp_spectrum_fromparam_ageDepMet_Q)

from fors2tostellarpopsynthesis.parameters import (SSPParametersFit,\
                                                   SSPParametersFitAgeDepMet,\
                                                   SSPParametersFit_AgeDepMet_Q,\
                                                   paramslist_to_dict)
from diffstar import sfh_singlegal

#get_ipython().run_line_magic('matplotlib', 'inline')
# to enlarge the sizes
params = {'legend.fontsize': 'large',
          'figure.figsize': (8, 4),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

import matplotlib.offsetbox
props = dict(boxstyle='round',edgecolor="w",facecolor="w", alpha=0.5)


# Load fors2tostellarpopsynthesis filters (GALEX, SDSS, VISTA)
ps = FilterInfo()

# Load LSST Filters
lsst_u = load_transmission_curve(fn="./DSPS_data/filters/lsst_u_transmission.h5")
lsst_g = load_transmission_curve(fn="./DSPS_data/filters/lsst_g_transmission.h5")
lsst_r = load_transmission_curve(fn="./DSPS_data/filters/lsst_r_transmission.h5")
lsst_i = load_transmission_curve(fn="./DSPS_data/filters/lsst_i_transmission.h5")
lsst_z = load_transmission_curve(fn="./DSPS_data/filters/lsst_z_transmission.h5")
lsst_y = load_transmission_curve(fn="./DSPS_data/filters/lsst_y_transmission.h5")

lsst_filters = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
lsst_cols = ['purple', 'blue', 'green', 'yellow', 'red', 'grey']


# Initial parameters for SPS
p = SSPParametersFit_AgeDepMet_Q()
init_params = p.INIT_PARAMS
params_min = p.PARAMS_MIN
params_max = p.PARAMS_MAX

params = {name:init_params[k] for k, name in enumerate(p.PARAM_NAMES_FLAT)}
min_params = {name:params_min[k] for k, name in enumerate(p.PARAM_NAMES_FLAT)}
max_params = {name:params_max[k] for k, name in enumerate(p.PARAM_NAMES_FLAT)}

@jit
def ssp_spectrum_fromparam_fullSFH(params, z_obs):
    """ Return the SED of SSP DSPS with original wavelength range wihout and with dust

    :param params: parameters for the fit
    :type params: dictionnary of parameters

    :param z_obs: redshift at which the model SSP should be calculated
    :type z_obs: float

    :return: the wavelength and the spectrum with dust and no dust
    :rtype: float

    """

    # decode the parameters
    MAH_lgmO = params["MAH_lgmO"]
    MAH_logtc = params["MAH_logtc"]
    MAH_early_index = params["MAH_early_index"]
    MAH_late_index = params["MAH_late_index"]
    list_param_mah = [MAH_lgmO,MAH_logtc,MAH_early_index,MAH_late_index]

    MS_lgmcrit = params["MS_lgmcrit"]
    MS_lgy_at_mcrit = params["MS_lgy_at_mcrit"]
    MS_indx_lo = params["MS_indx_lo"]
    MS_indx_hi = params["MS_indx_hi"]
    MS_tau_dep = params["MS_tau_dep"]
    list_param_ms = [MS_lgmcrit,MS_lgy_at_mcrit,MS_indx_lo,MS_indx_hi,MS_tau_dep]

    Q_lg_qt = params["Q_lg_qt"]
    Q_qlglgdt = params["Q_qlglgdt"]
    Q_lg_drop = params["Q_lg_drop"]
    Q_lg_rejuv = params["Q_lg_rejuv"]
    list_param_q = [Q_lg_qt, Q_qlglgdt,Q_lg_drop,Q_lg_rejuv]

    # compute SFR
    tarr = np.linspace(0.1, TODAY_GYR, 100)
    sfh_gal = sfh_singlegal(tarr, list_param_mah , list_param_ms, list_param_q)
    
    # Dust
    Av = params["AV"]
    uv_bump = params["UV_BUMP"]
    plaw_slope = params["PLAW_SLOPE"]
    list_param_dust = [Av,uv_bump,plaw_slope]

    # metallicity
    gal_lgmet = params["LGMET"] # log10(Z)
    gal_lgmet_scatter = params["LGMETSCATTER"]  # lognormal scatter in the metallicity distribution function

    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
    t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument

    # clear sfh in future
    #sfh_gal = jnp.where(tarr<t_obs, sfh_gal, 0)

    # compute the SED_info object
    gal_t_table = tarr
    gal_sfr_table = sfh_gal
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(gal_t_table, gal_sfr_table,\
                                                     gal_lgmet, gal_lgmet_scatter,\
                                                     SSP_DATA.ssp_lgmet, SSP_DATA.ssp_lg_age_gyr, SSP_DATA.ssp_flux,\
                                                     t_obs)

    # compute dust attenuation
    wave_spec_micron = SSP_DATA.ssp_wave/10_000
    k = sbl18_k_lambda(wave_spec_micron,uv_bump,plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k,Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    return SSP_DATA.ssp_wave, sed_info.rest_sed, sed_attenuated


# Dash app layout
app = Dash()

layout_SPS = []
for _par in params.keys():
    _mrk = { _v: f"{_v:.2f}" for _v in np.linspace(min_params[_par], max_params[_par], 5) }
    layout_SPS.append(html.Div([html.P(_par),\
                                html.Div([dcc.Slider(id=_par+'-value',\
                                                     min=min_params[_par], max=max_params[_par], value=params[_par],\
                                                     step=abs(max_params[_par]-min_params[_par])/10,\
                                                     marks=_mrk,\
                                                     tooltip={"placement":"bottom", "always_visible":True} )],\
                                         id=_par+"-slider")],\
                               style={'width': '25%', 'display': 'inline-block'}\
                              )
                     )

app.layout = html.Div([\
                       html.H1(f'Interactive plot of DSPS-synthesized SED: '),\
                       html.Div([dcc.Graph(id="SED")],\
                                style={'width': '50%', 'display': 'inline-block'}\
                               ),\
                       html.Div([dcc.Graph(id="SFR")],\
                                style={'width': '50%', 'display': 'inline-block'}\
                               ),\
                       html.Div([html.P("Redshift"),\
                                 html.Div([dcc.Slider(id='redshift-value', min=0., max=3., value=0.5, step=0.001,\
                                                      marks={0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2', 2.5:'2.5', 3:'3'},\
                                                      tooltip={"placement":"bottom", "always_visible":True} )],\
                                          id="z-slider")],\
                                 style={'width': '25%', 'display': 'inline-block'}\
                                )
                      ]+layout_SPS)

# Dash app run
@app.callback([Output("z-slider", "children")] + [Output(_par+"-slider", "children")\
                                                    for _par in params.keys()] + [Output("SED", "figure"),\
                                                                                  Output("SFR", "figure")],\
              [Input("redshift-value", "value")] + [Input(_par+"-value", "value")\
                                               for _par in params.keys()]\
             )
def display_graph(z, *sps_params):
    slid_z = dcc.Slider(id='redshift-value', min=0., max=3., value=z, step=0.001,\
                        marks={0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2', 2.5:'2.5', 3:'3'},\
                        tooltip={"placement":"bottom", "always_visible":True}\
                       )
    
    return_slids = []
    for _par, _par_val in zip(params.keys(), sps_params):
        _mrk = { _v: f"{_v:.2f}" for _v in np.linspace(min_params[_par], max_params[_par], 5) }
        return_slids.append(dcc.Slider(id=_par+'-value',\
                                       min=min_params[_par], max=max_params[_par], value=_par_val,\
                                       step=abs(max_params[_par]-min_params[_par])/10,\
                                       marks=_mrk,\
                                       tooltip={"placement":"bottom", "always_visible":True} )\
                           )
             
    sed_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y":True}]], shared_xaxes=True)
    sfr_fig = make_subplots(rows=1, cols=1)
    
    param_dict = {_p:_v for _p, _v in zip(params.keys(), sps_params)}
    
    sps_wls, sps_rest_sed, sps_attenuated_sed = ssp_spectrum_fromparam_ageDepMet_Q(param_dict, z)
    
    obs_mags = np.array([ calc_obs_mag(sps_wls, sps_attenuated_sed, filt.wave, filt.transmission,\
                                       z, *DEFAULT_COSMOLOGY) for filt in lsst_filters ])
    
    for omag, filt, col in zip(obs_mags, lsst_filters, lsst_cols):
        sed_fig.add_scatter(x=filt.wave, y=filt.transmission, mode='lines', line_color=col, secondary_y=False)
        #sed_fig.add_scatter(x=[np.median(filt.wave)], y=[omag], mode='markers', line_color=col, secondary_y=True)
    
    wls = (1+z)*sps_wls
    
    _sel = (wls > 1000.) * (wls < 12000.)
    AB_norm = 1.13492e-13*(np.log(wls[_sel][-1])-np.log(wls[_sel][0]))
    _selplot = (wls < 12000.)
    #sed_fig.add_scatter(x=wls[_selplot], y=-2.5*np.log10(sps_attenuated_sed[_selplot]/AB_norm), mode='lines', line_color='black', secondary_y=True)
    sed_fig.add_scatter(x=wls[_selplot], y=sps_attenuated_sed[_selplot],\
                        mode='lines', line_color='red',\
                        secondary_y=True)
    sed_fig.add_scatter(x=wls[_selplot], y=sps_rest_sed[_selplot],\
                        mode='lines', line_color='blue',\
                        secondary_y=True)
    #sed_fig.update_layout(yaxis2_type="log")
    t_obs, tarr, sfh_gal = mean_sfr_ageDepMet_Q(param_dict, z)
    sfr_fig.add_scatter(x=tarr, y=sfh_gal, mode='lines', line_color='black')
    sfr_fig.add_vline(x=t_obs, line_width=2, line_dash="dash", line_color="orange")
    
    return (slid_z, *return_slids, sed_fig, sfr_fig)

if __name__ == '__main__':
    app.run_server(debug=False)
