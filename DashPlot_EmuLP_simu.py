#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys, os, copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import n_colors
from dash import Dash, dcc, html, Input, Output

from dsps import load_ssp_templates
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY
from dsps import calc_rest_sed_sfh_table_lognormal_mdf
from dsps import calc_rest_sed_sfh_table_met_table
from dsps import load_transmission_curve
from dsps import calc_rest_mag
from dsps import calc_obs_mag
from dsps.data_loaders.defaults import TransmissionCurve

import pandas as pd
import pickle

from EmuLP import Cosmology, Filter, Galaxy, Estimator, Extinction, Template

#get_ipython().run_line_magic('matplotlib', 'inline')
# to enlarge the sizes
params = {'legend.fontsize': 'large',
          'figure.figsize': (20, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

import matplotlib.offsetbox
props = dict(boxstyle='round',edgecolor="w",facecolor="w", alpha=0.5)

# DSPS Params - kept for provision
## Load LSST Filters
lsst_u = load_transmission_curve(fn="./DSPS_data/filters/lsst_u_transmission.h5")
lsst_g = load_transmission_curve(fn="./DSPS_data/filters/lsst_g_transmission.h5")
lsst_r = load_transmission_curve(fn="./DSPS_data/filters/lsst_r_transmission.h5")
lsst_i = load_transmission_curve(fn="./DSPS_data/filters/lsst_i_transmission.h5")
lsst_z = load_transmission_curve(fn="./DSPS_data/filters/lsst_z_transmission.h5")
lsst_y = load_transmission_curve(fn="./DSPS_data/filters/lsst_y_transmission.h5")

lsst_filters = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
lsst_cols = ['purple', 'blue', 'green', 'yellow', 'red', 'grey']


## Load COSMOS Filters if available?


## Load SSP
ssp_data = load_ssp_templates(fn="./DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5")

## Parametrize SED-synthesis
gal_t_table = np.linspace(0.05, 13.8, 100) # age of the universe in Gyr
gal_sfr_table = np.random.uniform(0, 10, gal_t_table.size) # SFR in Msun/yr

gal_lgmet = -2.0 # log10(Z)
gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

gal_lgmet_table = np.linspace(-3, -2, gal_t_table.size)


# EmuLP configuration
'''
if len(args) > 1:
    conf_json = args[1] # le premier argument de args est toujours `__main__.py`
else :
    conf_json = 'EmuLP/defaults.json' # attention à la localisation du fichier !
'''
conf_json = 'EmuLP/COSMOS2020-with-FORS2-HSC_only.json'
with open(conf_json, "r") as inpfile:
    inputs = json.load(inpfile)

## Cosmology
lcdm = Cosmology.Cosmology(inputs['Cosmology']['h0'], inputs['Cosmology']['om0'], inputs['Cosmology']['l0'])

## Templates names and tables
templates_dict = inputs['Templates']
MODs=np.array([_k for _k in templates_dict.keys()])

## Grids - might not be useful here
z_grid = np.arange(inputs['Z_GRID']['z_min'],\
                   inputs['Z_GRID']['z_max']+inputs['Z_GRID']['z_step'],\
                   inputs['Z_GRID']['z_step'])
ZPs = z_grid

wl_grid = np.arange(inputs['WL_GRID']['lambda_min'],\
                    inputs['WL_GRID']['lambda_max']+inputs['WL_GRID']['lambda_step'],\
                    inputs['WL_GRID']['lambda_step'])

## Extinctions
extlaws_dict = inputs['Extinctions']
#extlaws_arr = np.array([Extinction.Extinction(_ext, extlaws_dict[_ext]) for _ext in extlaws_dict])
EXTs = np.array( [_k for _k in extlaws_dict.keys()] )
EBVs = np.array(inputs['e_BV'])

## Filters
filters_dict = inputs['Filters']
filters_arr = np.array( [ Filter.Filter(key, filters_dict[key]["path"], filters_dict[key]["transmission"])\
                         for key in filters_dict.keys() ] )
N_FILT = len(filters_arr)

cosmos_filters = [ TransmissionCurve(_filt.wavelengths, _filt.transmit) for _filt in filters_arr ]
#cosmos_cols = n_colors((0, 'purple'), (N_FILT-1, 'darkred'), N_FILT, colortype='tuple')
cosmos_cols = px.colors.sample_colorscale("rainbow", [n/(N_FILT -1) for n in range(N_FILT)])


## Test data
data_path = os.path.abspath(inputs['Dataset']['path'])
data_ismag = (inputs['Dataset']['type'].lower() == 'm')
data_file_arr = np.loadtxt(data_path)
GALs = data_file_arr[:,0]
_ex_ = GALs[0]


# Dash app layout
app = Dash()
app.layout = html.Div([
    html.H1(f'Interactive plot of EmuLP run for a single case: '),
    html.P("Test galaxy"),
    dcc.Dropdown(id='gal-id', options=GALs, value=_ex_),
    html.P("Template"),
    dcc.Dropdown(id='mod-id', options=MODs, value=MODs[0]),
    html.P("Extinction"),
    dcc.Dropdown(id='extinction', options=EXTs, value=EXTs[0]),
    html.P("E(B-V)"),
    dcc.Dropdown(id='EBV', options=EBVs, value=EBVs[0]),
    html.P("Estimator"),
    dcc.Dropdown(id='estimator', options=['chi2'], value='chi2'),
    dcc.Graph(id="plot-chi"),
    dcc.Graph(id="hist-chi"),
    dcc.Graph(id="plot-mod"),
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
@app.callback(\
              Output("z-slider", "children"),\
              Output("SFR-slider", "children"),\
              Output("met-slider", "children"),\
              Output("plot-chi", "figure"),\
              Output("hist-chi", "figure"),\
              Output("plot-mod", "figure"),\
              Input("redshift-value", "value"),\
              Input("sfr-value", "value"),\
              Input("met-value", "value"),\
              Input("gal-id", "value"),\
              Input("mod-id", "value"),\
              Input("extinction", "value"),\
              Input("EBV", "value"),\
              Input("estimator", "value")
             )
def display_graph(z, sfr, met, gal_id, mod_id, extinc, e_BV, estim_method):
    slid_z = dcc.Slider( id='redshift-value', min=0., max=3., value=z, step=0.001, marks={0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2', 2.5:'2.5', 3:'3'},\
                      tooltip={"placement":"bottom", "always_visible":True} )
    slid_sfr = dcc.Slider( id='sfr-value', min=0., max=10., value=sfr, step=0.01, marks={0: '0', 1:'1', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10'},\
                tooltip={"placement":"bottom", "always_visible":True} )
    slid_met = dcc.Slider( id='met-value', min=2, max=3, value=met, step=0.001, marks={2: '2', 2.2:'2.2', 2.4:'2.4', 2.6:'2.6', 2.8:'2.8', 3:'3'},\
                tooltip={"placement":"bottom", "always_visible":True} )
    
    plotChi = make_subplots(rows=1, cols=1, specs=[[{"secondary_y":False}]], shared_xaxes=False)
    
    # EmuLP embedded run
    templates_arr = np.array([])
    extinction = Extinction.Extinction(extinc, extlaws_dict[extinc])
    for _e_BV in EBVs:
        for redshift in z_grid:
            template = Template.Template(mod_id, templates_dict[mod_id])
            template.apply_extinc(extinction, _e_BV)
            template.to_redshift(redshift, lcdm)
            template.normalize(1000., 10000.)
            #print(np.trapz(filters_arr[0].f_transmit(template.wavelengths), template.wavelengths))
            template.fill_magnitudes(filters_arr)
            #print(template.magnitudes)
            templates_arr = np.append(templates_arr, copy.deepcopy(template))
            del template
                    
    if estim_method.lower() == 'chi2':
        zp_estim = Estimator.Chi2(estim_method, templates_arr)
    else:
        raise RuntimeError(f"Unimplemented estimator {estim_method}.\nPlease specify one of the following: chi2, <more to come>.")
    
    galaxy_arr = np.array([])
    
    for i in np.nonzero(GALs == gal_id)[0]:
        try:
            assert (len(data_file_arr[i,:]) == 1+2*N_FILT) or (len(data_file_arr[i,:]) == 1+2*N_FILT+1), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            if (len(data_file_arr[i,:]) == 1+2*N_FILT+1):
                gal = Galaxy.Galaxy(data_file_arr[i, 0], data_file_arr[i, 1:2*N_FILT+1], data_ismag, zs=data_file_arr[i, 2*N_FILT+1])
            else:
                gal = Galaxy.Galaxy(data_file_arr[i, 0], data_file_arr[i, 1:2*N_FILT+1], data_ismag)
            gal.estimate_zp(zp_estim, filters_arr)
            galaxy_arr = np.append(galaxy_arr, gal)
        except AssertionError:
            pass
    
    # Results in a dictionary
    results_dict = { int(float(_gal.id)): _gal.results_dict for _gal in galaxy_arr }
    zs_dict = { int(float(_gal.id)): _gal.zs for _gal in galaxy_arr }
    
    # Graphe chi² de EmuLP
    test_gal_df=pd.DataFrame(results_dict[gal_id])
    _df_ebv = test_gal_df[test_gal_df['eBV']==e_BV]
    chi_arr_allMods = np.column_stack(\
                                      [_df_ebv[_df_ebv['mod id']==mod]['chi2'].values\
                                       for mod in np.unique(_df_ebv['mod id'].values)\
                                      ]\
                                     ) 
    chi_avg_mods = np.average(chi_arr_allMods, axis=1)
    chi_std_mods = np.std(chi_arr_allMods, axis=1)
    
    
    _df_mod = test_gal_df[test_gal_df['mod id']==mod_id]
    chi_arr_allEBV = np.column_stack(\
                                     [_df_mod[_df_mod['eBV']==_ebv]['chi2'].values\
                                      for _ebv in np.unique(_df_mod['eBV'].values)\
                                     ]\
                                 )
    chi_avg_EBVs = np.average(chi_arr_allEBV, axis=1)
    chi_std_EBVs = np.std(chi_arr_allEBV, axis=1)
    
    _df_sel = _df_ebv[_df_ebv['mod id']==mod_id]
    plotChi.add_scatter(x=ZPs, y=chi_avg_mods, mode='lines', row=1, col=1,\
                        line_color='black', name='averaged over models')
    plotChi.add_scatter(x=ZPs, y=chi_avg_EBVs, mode='lines', row=1, col=1,\
                        line_color='blue', name='averaged over E(B-V)')
    plotChi.add_scatter(x=ZPs, y=_df_sel['chi2'].values, mode='lines', row=1, col=1,\
                        line_color='purple', name='Selected model')
    
    plotChi.add_trace(go.Scatter(x=np.concatenate([ZPs, ZPs[::-1]]),\
                             y=np.concatenate([chi_avg_mods+chi_std_mods, chi_avg_mods[::-1]-chi_std_mods[::-1]]),\
                             fill='toself', hoveron='points', line_color='grey', mode='lines'\
                            ),\
                  row=1, col=1\
                 )
    
    plotChi.add_trace(go.Scatter(x=np.concatenate([ZPs, ZPs[::-1]]),\
                             y=np.concatenate([chi_avg_EBVs+chi_std_EBVs, chi_avg_EBVs[::-1]-chi_std_EBVs[::-1]]),\
                             fill='toself', hoveron='points', line_color='cyan'\
                            ),\
                  row=1, col=1\
                 )
    
    '''
    for _prop in fig.layout:
        if _prop[:5] == 'yaxis':
            fig.layout[_prop]['type'] = 'log'
    '''
            
    plotChi.layout.yaxis1.type = 'log'
    plotChi.layout.height = 600
    
    plotChi.add_vline(x=zs_dict[gal_id],row=1, col=1, name='True z', line_color="green")
    plotChi.add_vline(x=ZPs[np.nanargmin(chi_avg_mods)],row=1, col=1, name='marg. over templates', line_color="red")
    plotChi.add_vline(x=ZPs[np.nanargmin(chi_avg_mods-chi_std_mods)],row=1, col=1,\
                  name='marg. over templates - $\sigma$', line_color="red", line_dash='dash')
    plotChi.add_vline(x=ZPs[np.nanargmin(chi_avg_EBVs)], row=1, col=1, name='marg. over E(B-V)', line_color="orange")
    plotChi.add_vline(x=ZPs[np.nanargmin(chi_avg_EBVs-chi_std_EBVs)], row=1, col=1,\
                  name='marg. over E(B-V) - $\sigma$', line_color="orange", line_dash='dash')
    
    plotChi.add_vline(x=ZPs[np.nanargmin(_df_sel['chi2'].values)],row=1, col=1,\
                  name='min. $\chi^2$ for selected conf.', line_color="purple")
    
    # Heatmap de chi²
    print(ZPs.shape, EBVs.shape, chi_arr_allEBV.shape)
    #histChi = make_subplots(rows=1, cols=1, specs=[[{"secondary_y":False}]], shared_xaxes=False)
    #histChi = go.Figure(go.Histogram2dContour(x=ZPs_bins, y=EBVs_bins, z=chi_arr_allEBV, histfunc="avg", colorscale="blues"))
    #histChi = go.Figure(go.Scatter(x=ZPs, y=EBVs, marker_color=chi_arr_allEBV, mode='markers', colorscale="blues"))
    histChi = px.scatter(_df_mod, x='zp', y='eBV', color='chi2',\
                         range_color=[0.95*np.min(_df_mod['chi2'].values),\
                                      np.median(_df_mod['chi2'].values)]\
                        )
    histChi.layout.height = 600
    
    # Représentation de la template, sa photométrie et celle de la galaxie test + DSPS
    template = Template.Template(mod_id, templates_dict[mod_id])
    template.apply_cosmo(lcdm, 0.)
    extinc_func = extinction.extinct_func(wl_grid, e_BV)
    temp_ext = copy.deepcopy(template)
    temp_ext.apply_extinc(extinction, e_BV)
    template.apply_cosmo(lcdm, 0.)
    temp_z = copy.deepcopy(template)
    temp_ext_z = copy.deepcopy(temp_ext)
    temp_z.to_redshift(z, lcdm)
    temp_ext_z.to_redshift(z, lcdm)
    #template.normalize(1000., 10000.)
    #print(np.trapz(filters_arr[0].f_transmit(template.wavelengths), template.wavelengths))
    template.fill_magnitudes(filters_arr)
    temp_ext.fill_magnitudes(filters_arr)
    temp_z.fill_magnitudes(filters_arr)
    temp_ext_z.fill_magnitudes(filters_arr)
    
    
    t_obs = age_at_z(z, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
    t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument
    
    ## Parametrize SED-synthesis
    gal_t_table = np.linspace(t_obs-0.01, t_obs+0.01, 100) # age of the universe in Gyr
    gal_sfr_table = np.full(gal_t_table.size, sfr) # SFR in Msun/yr

    gal_lgmet = -met # log10(Z)
    gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

    gal_lgmet_table = np.linspace(-3, -2, gal_t_table.size)
    
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(gal_t_table, gal_sfr_table, gal_lgmet, gal_lgmet_scatter,\
                                                     ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)
                                                     
    #sed_info = calc_rest_sed_sfh_table_met_table(gal_t_table, gal_sfr_table, gal_lgmet_table, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs)
    
    obs_mags = np.array([ calc_obs_mag(ssp_data.ssp_wave, sed_info.rest_sed, filt.wave, filt.transmission,\
                                       z, *DEFAULT_COSMOLOGY) for filt in cosmos_filters ])
    
    plotMod = make_subplots(rows=1, cols=1, specs=[[{"secondary_y":True}]], shared_xaxes=False)
    for omag, filt, col in zip(obs_mags, cosmos_filters, cosmos_cols):
        plotMod.add_scatter(x=filt.wave, y=filt.transmission,\
                        row=1, col=1,\
                        mode='lines', line_color=col, secondary_y=False, showlegend=False)
        plotMod.add_scatter(x=[np.median(filt.wave)], y=[omag],\
                        row=1, col=1,\
                        mode='markers', line_color=col, secondary_y=True, showlegend=False)
    
    wls = (1+z)*ssp_data.ssp_wave
    
    _sel = (wls > 1000.) * (wls < 11000.)
    AB_norm = 1.13492e-13*(np.log(wls[_sel][-1])-np.log(wls[_sel][0]))
    plotMod.add_scatter(x=wls[_sel], y=-2.5*np.log10(sed_info.rest_sed[_sel]/AB_norm),\
                    row=1, col=1,\
                    mode='lines', line_color='black', secondary_y=True)
    
    _sel1 = (template.wavelengths > 1000.) * (template.wavelengths < 11000.)
    AB_norm1 = 1.13492e-13*(np.log(template.wavelengths[_sel1][-1])-np.log(template.wavelengths[_sel1][0]))
    plotMod.add_scatter(x=template.wavelengths[_sel1], y=-2.5*np.log10(template.lumins[_sel1]/AB_norm1),\
                    row=1, col=1,\
                    mode='lines', line_color='blue', secondary_y=True)
    
    _sel2 = (temp_ext.wavelengths > 1000.) * (temp_ext.wavelengths < 11000.)
    AB_norm2 = 1.13492e-13*(np.log(temp_ext.wavelengths[_sel2][-1])-np.log(temp_ext.wavelengths[_sel2][0]))
    plotMod.add_scatter(x=temp_ext.wavelengths[_sel2], y=-2.5*np.log10(temp_ext.lumins[_sel2]/AB_norm2),\
                    row=1, col=1,\
                    mode='lines', line_color='purple', secondary_y=True)
    
    _sel3 = (temp_z.wavelengths > 1000.) * (temp_z.wavelengths < 11000.)
    AB_norm3 = 1.13492e-13*(np.log(temp_z.wavelengths[_sel3][-1])-np.log(temp_z.wavelengths[_sel3][0]))
    plotMod.add_scatter(x=temp_z.wavelengths[_sel3], y=-2.5*np.log10(temp_z.lumins[_sel3]/AB_norm3),\
                    row=1, col=1,\
                    mode='lines', line_color='red', secondary_y=True)
    
    _sel4 = (temp_ext_z.wavelengths > 1000.) * (temp_ext_z.wavelengths < 11000.)
    AB_norm4 = 1.13492e-13*(np.log(temp_ext_z.wavelengths[_sel4][-1])-np.log(temp_ext_z.wavelengths[_sel4][0]))
    plotMod.add_scatter(x=temp_ext_z.wavelengths[_sel4], y=-2.5*np.log10(temp_ext_z.lumins[_sel4]/AB_norm4),\
                    row=1, col=1,\
                    mode='lines', line_color='darkred', secondary_y=True)
    
    plotMod.layout.height = 600
    
    return slid_z, slid_sfr, slid_met, plotChi, histChi, plotMod

if __name__ == '__main__':
    app.run_server(debug=False)
