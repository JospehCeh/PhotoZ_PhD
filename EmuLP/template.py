#!/bin/env python3
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os, sys

class Template:
	"""SED Templates to be used for photo-z estimation"""
	
	def __init__(self, specfile):
		self.path = specfile
		_wl, _lum = np.loadtxt(specfile, unpack=True)
		f_lum = interp1d(_wl, _lum, bounds_error=False, fill_value=0.)
		self.wavelengths = np.arange(100., 60000., 10.)
		self.lumins = f_lum(self.wavelengths)
		self.e_bv = 0.
		self.extinc_law = 0 # 0 = none, 1 = Calzetti, 2 = Prevot
		self.redshift = 0.

	def normalize(self, wl_inf, wl_sup):
		self.norm = np.trapz(self.lumin, wl_inf, wl_sup)
		self.spec_norm = self.lumin / self.norm

	def to_df(self):
		df = pd.DataFrame()
		return df
	
	def to_redshift(self, z):
		self.wavelengths = self.wavelengths * (1.+z)
		self.redshift = z
	
	def to_restframe(self):
		self.wavelengths = self.wavelengths / (1.+self.redshift)
		self.redshift = 0.
		
	def apply_extinc(law, e_bv):
		if law == 1:
			# voir LePhare
		elif law == 2:
			# voir LePhare
			
	def compute_flux(self, filt):
		flux = np.trapz(self.wavelengths, self.lumins * filt.transmit)/(4*np.pi*10**2)
		return flux
	
	def compute_magAB(self, filt):
		flux = self.compute_flux(filt)
		mag = -2.5*np.log10(flux) + 48.6
			
	
		

