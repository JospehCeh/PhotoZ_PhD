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

# Load LSST Filters
lsst_u = load_transmission_curve(fn="./DSPS_data/filters/lsst_u_transmission.h5")
lsst_g = load_transmission_curve(fn="./DSPS_data/filters/lsst_g_transmission.h5")
lsst_r = load_transmission_curve(fn="./DSPS_data/filters/lsst_r_transmission.h5")
lsst_i = load_transmission_curve(fn="./DSPS_data/filters/lsst_i_transmission.h5")
lsst_z = load_transmission_curve(fn="./DSPS_data/filters/lsst_z_transmission.h5")
lsst_y = load_transmission_curve(fn="./DSPS_data/filters/lsst_y_transmission.h5")

lsst_filters = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
lsst_cols = ['purple', 'blue', 'green', 'yellow', 'red', 'grey']

# Load SSP
ssp_data = load_ssp_templates(fn="./DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5")

# Parametrize SED-synthesis
gal_t_table = np.linspace(0.05, 13.8, 100) # age of the universe in Gyr
gal_sfr_table = np.random.uniform(0, 10, gal_t_table.size) # SFR in Msun/yr

gal_lgmet = -2.0 # log10(Z)
gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

gal_lgmet_table = np.linspace(-3, -2, gal_t_table.size)

#z_obs = np.linspace(0.01, 3.01, 100)
#t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
#t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument

#sed_info = calc_rest_sed_sfh_table_lognormal_mdf(gal_t_table, gal_sfr_table, gal_lgmet, gal_lgmet_scatter,\
#                                                 ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs[0])

#sed_info = calc_rest_sed_sfh_table_met_table(gal_t_table, gal_sfr_table, gal_lgmet_table, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)


# Dash app layout
app = Dash()
app.layout = html.Div([
    html.H1(f'Interactive plot of DSPS-synthesized SED: '),
    dcc.Graph(id="graph"),
    html.P("Redshift"),
    html.Div([dcc.Slider( id='redshift-value', min=0., max=3., value=0.5, step=0.001, marks={0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2', 2.5:'2.5', 3:'3'},\
                tooltip={"placement":"bottom", "always_visible":True} )], id="z-slider"),
    html.P("Star Formation Rate"),
    html.Div([dcc.Slider( id='sfr-value', min=0., max=10., value=5, step=0.01, marks={0: '0', 1:'1', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10'},\
                tooltip={"placement":"bottom", "always_visible":True} )], id="SFR-slider"),
    html.P("Metallicity"),
    html.Div([dcc.Slider( id='met-value', min=2, max=3, value=2, step=0.001, marks={2: '2', 2.2:'2.2', 2.4:'2.4', 2.6:'2.6', 2.8:'2.8', 3:'3'},\
                tooltip={"placement":"bottom", "always_visible":True} )], id="met-slider")
])

# Dash app run
@app.callback( Output("z-slider", "children"), Output("SFR-slider", "children"), Output("met-slider", "children"), Output("graph", "figure"), Input("redshift-value", "value"), Input("sfr-value", "value"), Input("met-value", "value"))
def display_graph(z, sfr, met):
    slid_z = dcc.Slider( id='redshift-value', min=0., max=3., value=z, step=0.001, marks={0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2', 2.5:'2.5', 3:'3'},\
                      tooltip={"placement":"bottom", "always_visible":True} )
    slid_sfr = dcc.Slider( id='sfr-value', min=0., max=10., value=sfr, step=0.01, marks={0: '0', 1:'1', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10'},\
                tooltip={"placement":"bottom", "always_visible":True} )
    slid_met = dcc.Slider( id='met-value', min=2, max=3, value=met, step=0.001, marks={2: '2', 2.2:'2.2', 2.4:'2.4', 2.6:'2.6', 2.8:'2.8', 3:'3'},\
                tooltip={"placement":"bottom", "always_visible":True} )
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y":True}]], shared_xaxes=True)
    
    t_obs = age_at_z(z, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
    t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument
    
    # Parametrize SED-synthesis
    gal_t_table = np.linspace(t_obs-0.01, t_obs+0.01, 100) # age of the universe in Gyr
    gal_sfr_table = np.full(gal_t_table.size, sfr) # SFR in Msun/yr

    gal_lgmet = -met # log10(Z)
    gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

    gal_lgmet_table = np.linspace(-3, -2, gal_t_table.size)
    
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(gal_t_table, gal_sfr_table, gal_lgmet, gal_lgmet_scatter,\
                                                     ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)
                                                     
    #sed_info = calc_rest_sed_sfh_table_met_table(gal_t_table, gal_sfr_table, gal_lgmet_table, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)
    
    obs_mags = np.array([ calc_obs_mag(ssp_data.ssp_wave, sed_info.rest_sed, filt.wave, filt.transmission,\
                                       z, *DEFAULT_COSMOLOGY) for filt in lsst_filters ])
    
    for omag, filt, col in zip(obs_mags, lsst_filters, lsst_cols):
        fig.add_scatter(x=filt.wave, y=filt.transmission, mode='lines', line_color=col, secondary_y=False)
        fig.add_scatter(x=[np.median(filt.wave)], y=[omag], mode='markers', line_color=col, secondary_y=True)
    
    wls = (1+z)*ssp_data.ssp_wave
    
    _sel = (wls > 1000.) * (wls < 12000.)
    AB_norm = 1.13492e-13*(np.log(wls[_sel][-1])-np.log(wls[_sel][0]))
    fig.add_scatter(x=wls[_sel], y=-2.5*np.log10(sed_info.rest_sed[_sel]/AB_norm), mode='lines', line_color='black', secondary_y=True)
    
    return slid_z, slid_sfr, slid_met, fig

if __name__ == '__main__':
    app.run_server(debug=False)
