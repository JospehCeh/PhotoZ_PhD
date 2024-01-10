#!/usr/bin/env python
# coding: utf-8
# # TD : The ACT measurement of the lensing power spectrum
# In[1]:


import numpy as np


# ## 1. Energy densities

# In[2]:


# 1.
H_0 = 67.5 #km/s/Mpc
mpc_m = 3.086e22 #m
H_0_s = H_0 / (mpc_m/1000.) # en s^{-1}


# In[3]:


# 2.
G = 6.674e-11 # m³/kg/s²
h = H_0/100 # adim.
Om_b = 0.022 / (h**2)
Om_cdm = 0.122 / (h**2)
rho_0_c = 3*H_0_s**2 / (8*np.pi*G) # kg/m³
rho_0_b = Om_b*rho_0_c
rho_0_cdm = Om_cdm*rho_0_c


# In[4]:


m_H = 1.6735575e-27 # kg
n_H = rho_0_b/m_H


# In[5]:


# 3.
# Sur feuille (calcul canalytique sans difficulté)


# In[6]:


# 4. - en admettant solution 3
kb = 1.380649e-23 # J/K
T_CMB = 2.7255 # K
c = 3.0e8 # km/s
h = 6.62607015e-34
hbar = h/(2*np.pi)
rho_0_g = (np.pi**2 * (kb*T_CMB)**4)/(15 * (hbar**3) * (c**5))
Om_g = rho_0_g / rho_0_c


# In[7]:


# 5.
N_eff = 3.046
rho_0_N = N_eff * (7/8) * (4/11)**(4/3) * rho_0_g
Om_N = rho_0_N / rho_0_c
Om_rad = Om_g+Om_N


# In[8]:


# 6.
Om_m = Om_b+Om_cdm
Om_L = 1. - Om_rad - Om_m


# ## 2. Distances

# In[9]:


# 1.
def H(z):
    return H_0_s*np.power(Om_rad*np.power((1+z),4)+Om_m*np.power((1+z),3)+Om_L, 0.5)

def integrand(z):
    return c/H(z)

z_cmb=1100.

from scipy.integrate import quad

def comoving_dist(z, Z):
    dist, err = quad(integrand, z, Z)
    return dist

dist_CMB = comoving_dist(0., z_cmb) # km
dist_CMB_Mpc = dist_CMB / mpc_m # Mpc


# In[10]:


ly = c * (365.25*24.*3600.) # km
dist_CMB_Gly = dist_CMB / (1.0e9*ly)


# La distance comobile entre nous et le CMB est plus de 3x supérieure à l'âge de l'Univers !!


