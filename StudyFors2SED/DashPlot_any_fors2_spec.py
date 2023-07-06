#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
import h5py
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pylick.analysis import Galaxy, Catalog
from pylick.indices import IndexLibrary
from dash import Dash, dcc, html, Input, Output

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

from def_raw_seds import *
from raw_data_analysis import *
from spectroscopy import *

t = Table.read(filename_fits_catalog)


# In[2]:
def build_emissionlinesdict():
    """  
    Build a dictionnary of lines in galaxies
    
    """
    
    
    #http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
    #https://classic.sdss.org/dr6/algorithms/linestable.html



    df_lines=pd.read_excel("datatools/GalEmissionLines.xlsx")
    df_sdss_lines = pd.read_excel("datatools/sdss_galaxylines.xlsx")
    
    lines_to_plot={}
    
    # K
    sel = df_sdss_lines["species"] == 'K'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["K"]={"wls":wls,"name":"K","type":"absorption"}
    
    # H
    sel = df_sdss_lines["species"] == 'H'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["H"]={"wls":wls,"name":"H","type":"absorption"}
    
    # G
    sel = df_sdss_lines["species"] == 'G'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["G"]={"wls":wls,"name":"G","type":"absorption"}
    
    # Mg
    sel = df_sdss_lines["species"] == 'Mg'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["Mg"]={"wls":wls,"name":"Mg","type":"absorption"}
    
    # Na
    sel = df_sdss_lines["species"] == 'Na'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["Na"]={"wls":wls,"name":"Na","type":"absorption"}
    
    # H8
    sel = df_lines["ion"] == 'H8'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H8"]={"wls":wls,"name":"$H8$","type":"emission"}
    
    # H9
    sel = df_lines["ion"] == 'H9'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H9"]={"wls":wls,"name":"$H9$","type":"emission"}
    
    # H10
    sel = df_lines["ion"] == 'H10'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H10"]={"wls":wls,"name":"$H10$","type":"emission"}
    
    # H11
    sel = df_lines["ion"] == 'H11'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H11"]={"wls":wls,"name":"$H11$","type":"emission"}
    
    # Halpha
    sel = df_lines["ion"] == 'Hα' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{alpha}"]={"wls":wls,"name":"$H_\\alpha$","type":"emission"}
    
    
    # Hbeta
    sel = df_lines["ion"] == 'Hβ' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{beta}"]={"wls":wls,"name":"$H_\\beta$","type":"emission"}

    # Hgamma
    sel = df_lines["ion"] == 'Hγ' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{gamma}"]={"wls":wls,"name":"$H_\\gamma$","type":"emission"}
    
    # Hdelta
    sel = df_lines["ion"] == 'Hδ' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{delta}"]={"wls":wls,"name":"$H_\\delta$","type":"emission"}
    
    # Hepsilon
    sel = df_lines["ion"] == 'Hε' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{epsilon}"]={"wls":wls,"name":"$H_\\epsilon$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[O II]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OII]"]={"wls":wls,"name":"$[OII]$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[O III]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OIII]"]={"wls":wls,"name":"$[OIII]$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'O IV]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OIV]"]={"wls":wls,"name":"$[OIV]$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'O VI'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OVI]"]={"wls":wls,"name":"$[OVI]$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Mgb'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Mgb"]={"wls":wls,"name":"$Mgb$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Mg II]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["MgII"]={"wls":wls,"name":"$MgII$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Fe43'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Fe43"]={"wls":wls,"name":"$Fe43$","type":"emission"}
    
    sel = df_lines["ion"] == 'Fe45'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Fe45"]={"wls":wls,"name":"$Fe45$","type":"emission"}
    
    sel = df_lines["ion"] == 'Ca44'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ca44"]={"wls":wls,"name":"$Ca44$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Ca44'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ca44"]={"wls":wls,"name":"$Ca44$","type":"emission"}
    
    sel = df_lines["ion"] == 'E'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["E"]={"wls":wls,"name":"$E$","type":"emission"}
    
    sel = df_lines["ion"] =='Fe II'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["FeII24"]={"wls":wls,"name":"$FeII24$","type":"emission"}
    lines_to_plot['FeII26']={"wls":wls,"name":"$FeII26$","type":"emission"}
    
    
    lines_to_plot['weak']={"wls":[],"name":"$weak$","type":"break"}
    lines_to_plot['?']={"wls":[],"name":"$?$","type":"break"}
    
    lines_to_plot['4000{AA}-break']={"wls":[4000.],"name":"$Bal$","type":"break"}
     
    sel = df_lines["ion"] == 'Lyα'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{alpha}"]={"wls":wls,"name":"$Ly_\\alpha$","type":"emission"}
    
    sel = df_lines["ion"] == 'Lyβ'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{beta}"]={"wls":wls,"name":"$Ly_\\beta$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Lyδ'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{delta}"]={"wls":wls,"name":"$Ly_\\delta$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Lyε'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{epsilon}"]={"wls":wls,"name":"$Ly_\\epsilon$","type":"emission"}
    
    sel = df_lines["ion"] == 'C IV'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["CIV"]={"wls":wls,"name":"$CIV$","type":"emission"}
    
    sel = df_lines["ion"] == 'Al III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["AlIII"]={"wls":wls,"name":"$AlIII$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[Ne III]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['NeIII']={"wls":wls,"name":"$NeIII$","type":"emission"}
    
    sel = df_lines["ion"] == 'He I'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['HeI']={"wls":wls,"name":"$HeI$","type":"emission"}
    
    sel = df_lines["ion"] == 'N III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['NIII']={"wls":wls,"name":"$NIII$","type":"emission"}
    
    sel = df_lines["ion"] == 'Al II'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['AlII']={"wls":wls,"name":"$AlII$","type":"emission"}
    
    sel = df_lines["ion"] == 'Al III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['AlIII']={"wls":wls,"name":"$AlIII$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[N II]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['NII']={"wls":wls,"name":"$NII$","type":"emission"}
    
    sel = df_lines["ion"] == 'C III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['CIII']={"wls":wls,"name":"$CIII$","type":"emission"}
    
    sel = df_lines["ion"] == 'C IV'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['CIV']={"wls":wls,"name":"$CIV$","type":"emission"}
    
    sel = df_sdss_lines["species"] == 'Si IV + O IV'
    wls=df_sdss_lines[sel]["wl"].values
    lines_to_plot['SiIV/OIV']={"wls":wls,"name":"$SiIV/OIV$","type":"emission"}
    
    lines_to_plot["(QSO)"] = {"wls":[],"name":"$QSO$","type":"emission"}
    lines_to_plot["QSO"] = {"wls":[],"name":"$QSO$","type":"emission"}
    
    lines_to_plot['NaD'] = {"wls":[],"name":"$NaD$","type":"emission"}
    
    lines_to_plot['broad'] = {"wls":[],"name":"$broad$","type":"emission"}
    
    return lines_to_plot


# In[3]:
def GetColumnHfData(hff,list_of_keys,nameval):
    """
    Extract hff atttribute 
    
    parameters
      hff           : descriptor of h5 file
      list_of_keys : list of spectra names
      nameval      : name of the attribute
      
    return
           the array of values in the order of 
    """
    all_data = []
    for key in list_of_keys:
        group=hff.get(key)
        val=group.attrs[nameval]
        all_data.append(val)
    return all_data


# In[4]:
def ReadFors2h5FileAttributes(hf):
    hf =  h5py.File(input_file_h5, 'r') 
    list_of_keys = list(hf.keys())
    # pick one key    
    key_sel =  list_of_keys[0]
    # pick one group
    group = hf.get(key_sel)  
    #pickup all attribute names
    all_subgroup_keys = []
    for k in group.attrs.keys():
        all_subgroup_keys.append(k)
        
    #print(all_subgroup_keys)
    # create info
    df_info = pd.DataFrame()
    for key in all_subgroup_keys:
        arr=GetColumnHfData(hf, list_of_keys ,key)
        df_info[key] = arr
    df_info.sort_values(by="num", ascending=True,inplace=True)
    df_info_num = df_info["num"].values
    key_tags = [ f"SPEC{num}" for num in df_info_num ]
    df_info["name"] = key_tags
    
    #'Nsp', 'RT', 'RV', 'Rmag', 'dec', 'eRV', 'lines', 'num', 'ra', 'redshift',
    
    #df_info = df_info[['num' ,'name', 'ra', 'dec', 'Rmag','redshift','lines','RT','RV','eRV','Nsp']]
    return df_info


# In[5]:
input_file_h5  = '../../WIP_FORS2/PhotoZ_PhD/QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5'


# In[6]:
ordered_keys = ['name','num','ra','dec', 'redshift','Rmag','RT', 'RV','eRV','Nsp','lines',
                'ra_galex','dec_galex','fuv_mag', 'fuv_magerr','nuv_mag', 'nuv_magerr', 
                'fuv_flux', 'fuv_fluxerr','nuv_flux', 'nuv_fluxerr','asep_galex',
                'ID', 'KIDS_TILE','RAJ2000','DECJ2000','Z_ML', 'Z_B','asep_kids','CLASS_STAR',
                'MAG_GAAP_u','MAG_GAAP_g','MAG_GAAP_r','MAG_GAAP_i','MAG_GAAP_Z','MAG_GAAP_Y','MAG_GAAP_J', 'MAG_GAAP_H','MAG_GAAP_Ks',
                'MAGERR_GAAP_u','MAGERR_GAAP_g','MAGERR_GAAP_r','MAGERR_GAAP_i','MAGERR_GAAP_Z','MAGERR_GAAP_Y','MAGERR_GAAP_J','MAGERR_GAAP_H','MAGERR_GAAP_Ks',
                'FLUX_GAAP_u','FLUX_GAAP_g','FLUX_GAAP_r','FLUX_GAAP_i','FLUX_GAAP_Z','FLUX_GAAP_Y','FLUX_GAAP_J', 'FLUX_GAAP_H','FLUX_GAAP_Ks',
                'FLUXERR_GAAP_u','FLUXERR_GAAP_g','FLUXERR_GAAP_r','FLUXERR_GAAP_i','FLUXERR_GAAP_Z','FLUXERR_GAAP_Y','FLUXERR_GAAP_J','FLUXERR_GAAP_H','FLUXERR_GAAP_Ks',
                'FLUX_RADIUS', 'EXTINCTION_u','EXTINCTION_g','EXTINCTION_r', 'EXTINCTION_i',]


# In[7]:
hf =  h5py.File(input_file_h5, 'r') 
list_of_keys = list(hf.keys())
df_info = ReadFors2h5FileAttributes(hf)
df_info.reset_index(drop=True, inplace=True)
df_info = df_info[ordered_keys]


# In[8]:
lines_to_plot = build_emissionlinesdict()

# In[9]:
col_dict = {'emission' : 'green' , 'absorption': 'red', 'break': 'blue'}

## Essai avec pyLick
ids=np.array(t['ID'])
valid_ids = df_info['num'].values
all_lines=np.array([l.decode().split(' ')[0] for l in np.asarray(t['Lines'])])
indices_list = np.array([37, 38, 40, 41, 42, 45, 46, 53, 57, 67, 68])

id_marks = { f'{valid}':int(valid) for valid in valid_ids }

init_id = np.random.choice(valid_ids, size=1)[0]
z0=df_info[df_info['num']==init_id]['redshift'].values[0]

# In[20]:
fors2out_path=os.path.abspath(os.path.join('.', 'fors2out', 'seds'))
fors2seds_path=os.path.abspath(os.path.join('..', 'fors2', 'seds'))
starlight_path=os.path.abspath(os.path.join('..', 'ResStarlight', 'BC03N', 'conf1', 'HZ4', 'output_rebuild_BC', 'full_spectra'))
starlightExt_path=os.path.abspath(os.path.join('..', 'ResStarlight', 'BC03N', 'conf1', 'HZ4', 'output_rebuild_BC', 'full_spectra_ext'))

wls_interp = np.arange(800., 11000., 10.)
df_spectra = pd.DataFrame()
df_spectra["Wavelength"]=wls_interp
_dict_for_df = {}
for gal_id in valid_ids:
    fors2_file = os.path.join(fors2seds_path,f'SPEC{gal_id}n.txt')
    fors2out_file = os.path.join(fors2out_path,f'SPEC{gal_id}.txt')
    SLout_file = os.path.join(starlight_path,f'SPEC{gal_id}_HZ4_BC.txt')
    wls1, lums1, _dum = np.loadtxt(fors2_file, unpack=True)
    _finterp1 = interp1d(wls1, lums1, kind='linear', bounds_error=False, fill_value=0.)
    _lums = _finterp1(wls_interp)
    _errs = 0.01*_lums
    
    #wls2, lums2 = np.loadtxt(fors2out_file, unpack=True)
    #wls3, lums3 = np.loadtxt(SLout_file, unpack=True)
    #wls4, lums4 = np.loadtxt(SLout_file, unpack=True)
    _dict_for_df[f"SPEC{gal_id}"] = _lums
    _dict_for_df[f"err_SPEC{gal_id}"] = _errs
_df = pd.DataFrame(_dict_for_df)
df_spectra = pd.concat([df_spectra, _df], axis=1)

app = Dash()
app.layout = html.Div([
    html.H1(f'Interactive plot of spectrum ID and EQW of researched lines : '),
    html.Div(id='lines'),
    html.P("Galaxy ID"),
    dcc.Dropdown(id='gal-id', options=id_marks, value=init_id),
    dcc.Graph(id="graph"),
    html.P("Redshift"),
    html.Div([dcc.Slider( id='redshift-value', min=0., max=2., value=z0, step=0.001, marks={z0:f'{z0}', 0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2'},\
                tooltip={"placement":"bottom", "always_visible":True} )], id="z-slider")
    
])

@app.callback( Output("z-slider", "children"), Output("lines", "children"), Output("graph", "figure"), Input("redshift-value", "value"), Input("gal-id", "value") )
def display_graph(z, test_gal_id):
    z0 = df_info[df_info['num']==int(test_gal_id)]['redshift'].values[0]
    slid = dcc.Slider( id='redshift-value', min=0., max=2., value=z, step=0.001, marks={z0:f'{z0}', 0: '0', 0.5:'0.5', 1:'1', 1.5:'1.5', 2: '2'},\
                        tooltip={"placement":"bottom", "always_visible":True} )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    #test_gal_id = np.random.choice(df_info['num'].values, size=1, replace=False)[0
    _sel = np.where((ids==int(test_gal_id)))[0]
    _lines=all_lines[_sel]
    pylick_inds1 = Galaxy(f'{test_gal_id}', indices_list, df_spectra["Wavelength"].values, df_spectra[f"SPEC{test_gal_id}"].values,\
                            df_spectra[f"err_SPEC{test_gal_id}"].values,\
                          index_table='./table_FORS2.dat', meas_method='int', z=z,\
                          plot=False, plot_settings={}\
                         )
    lib=IndexLibrary(pylick_inds1.index_table, pylick_inds1.index_list)

    #df_spectra["Wavelength"] = df_spectra["Wavelength"].values/(1+z)
    wls = df_spectra["Wavelength"].values/(1+z)
    wls_int = np.arange(wls[0], wls[-1], 0.0625)
    lums = df_spectra[f"SPEC{test_gal_id}"].values # * np.power(wls, 2)

    spec_wls, spec_lums = np.array([]), np.array([])
    specInt_wls, specInt_lums, cont_lums = np.array([]), np.array([]), np.array([])

    for _loc, ind_lib in enumerate(indices_list):
        ew = pylick_inds1.vals[_loc]
        if np.isfinite(ew):
            bb, br = lib[[ind_lib]][ind_lib]['blue']
            cb, cr = lib[[ind_lib]][ind_lib]['centr']
            rb, rr = lib[[ind_lib]][ind_lib]['red']
            
            _selcont = ((wls>=bb) * (wls<=cb)) + ((wls>=cr) * (wls<=rr))
            _selband = (wls>=bb) * (wls<=rr)

            cont = interp1d(wls[_selcont], lums[_selcont], kind='zero', bounds_error=False, fill_value=(lums[_selband][0], lums[_selband][-1]))
            sed = interp1d(wls[_selband], lums[_selband], kind='zero', bounds_error=False, fill_value=(lums[_selband][0], lums[_selband][-1]))

            spec_wls = np.append(spec_wls, wls[_selband])
            spec_lums = np.append(spec_lums, lums[_selband]/cont(wls[_selband]))

            _selcont = ((wls_int>=bb) * (wls_int<=cb)) + ((wls_int>=cr) * (wls_int<=rr))
            _selband = (wls_int>=bb) * (wls_int<=rr)

            specInt_wls = np.append(specInt_wls, wls_int[_selband])
            specInt_lums = np.append(specInt_lums, sed(wls_int[_selband])/cont(wls_int[_selband]))
            cont_lums = np.append(cont_lums, cont(wls_int[_selband])/cont(wls_int[_selband]))

    fig.add_scatter(x=spec_wls, y=spec_lums, row=1, col=1, mode='markers+lines', name='Spectrum', line_color='blue')
    fig.add_scatter(x=specInt_wls, y=specInt_lums, row=1, col=1, mode='markers+lines', name='Interpolated spectrum', line_color='purple')
    fig.add_scatter(x=specInt_wls, y=cont_lums, row=1, col=1, mode='markers+lines', name='Continuum', line_color='grey')
    fig.add_scatter(x=wls, y=lums, name=f"SPEC{test_gal_id}", mode='lines', row=2, col=1, line_color='blue')
    
    for _loc, ind_lib in enumerate(indices_list):
        ew = pylick_inds1.vals[_loc]
        if np.isfinite(ew):
            bb, br = lib[[ind_lib]][ind_lib]['blue']
            cb, cr = lib[[ind_lib]][ind_lib]['centr']
            rb, rr = lib[[ind_lib]][ind_lib]['red']

            _color = col_dict['emission']
            if ew > 0:
                _color = col_dict['absorption']

            fig.add_vrect(x0=0.5*(cb+cr)-0.5*abs(ew), x1=0.5*(cb+cr)+0.5*abs(ew),\
                          annotation_text=f"EQW({lib[[ind_lib]][ind_lib]['name']})", annotation_position="top left",\
                          row='all', col='all', fillcolor=_color, opacity=0.25, line_width=0.1)
    return slid, html.Span(f"{_lines}"), fig

if __name__ == '__main__':
    app.run_server(debug=False)
