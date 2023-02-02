# Analysis of fors2 raw data builded by JCT and located in
# /home/enuss/00_labo/lsst/photoz/edmond_fors2/dataset_edmond/edmond_lib_jct/all_txt
# and sorted in 
# /home/enuss/00_labo/lsst/photoz/edmond_fors2/dataset_edmond/raw_SEDs/txt_jct/
# after cleaning and writen in stalight format
# 
# source ~/setmeup/photoz_setmeup.zsh; ipython raw_data_analysis.py

import os,sys
from matplotlib import pylab as plt
import numpy as np
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages
#from scipy import interp as scinterp
from scipy.interpolate import interp1d
from astropy.io import fits
import glob
import collections
from def_raw_seds import *
#-----------------------------------------------------

#noramlization for SL spectra:
delta=500.
b1=4150. #lower  bound for normalization
b2=4250. #higher bound for normalization

low_b =1500. #lower  bound for SL plot
high_b=8000. #higher bound for SL plot

#var_cut=0.002
var_cut=0.005

def plot_fig(x1,y1,col1,lab1,x2,y2,col2,lab2,title):
  #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
  fig = plt.figure(figsize=(12, 6), dpi=100)
  plt.plot(x1,y1,col1, linewidth = 2,label=lab1)
  plt.plot(x2,y2,col2, linewidth = 2,label=lab2)
  plt.xlabel(r'$\lambda$ [$\AA$]',size=18)
  plt.ylabel(r'Flux (arbitrary units)',size=18)
  plt.xlim(700,9000)
  plt.grid(True)
  plt.title(title)
  #plt.legend(loc=4); #bas droite
  #plt.legend(loc=3); #bas gauche
  #plt.legend(loc=2); #haut gauche
  plt.legend(loc=1); #haut droit
  pdf_pages.savefig(fig)
  
col1=['b-' ,'g-' ,'r-' ,'c-' ,'m-' ,'y-' ,'k-']

pdf_pages = PdfPages('all_plots.pdf')

norm=1.E-14
  
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# plot des spectres fors2 raw : :
def plot_fors2_raw():
    for i in lst:
      filename=path_out_jct+i+'.txt'
      if (os.path.exists(filename) != True):
         print(filename)
      #'''
      base=os.path.basename(i).split('.')[0]
      id=base[4:]
      spec_jct=SED_jct(id)

      spec=SED(filename)
      x=spec.wave
      if ((min(x)<3900)and(max(x)>3950)):
        y=spec.flux*norm/spec.get_scale(bounds=(3900,3950))
        print(('usual normalization',i))
      else:
        y=spec.flux*norm/abs(spec.get_scale(bounds=(min(x)+100,min(x)+150)))
        print(('specific normalization',i))
        
      #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
      fig = plt.figure(figsize=(12, 6), dpi=100)
      plt.semilogy(x,y,'r', linewidth = 2,label=i)
      plt.xlabel(r'$\lambda$ [$\AA$]',size=18)
      plt.ylabel(r'Flux (arbitrary units)',size=18)
      plt.ylim(5e-16,9e-14)
      if (i=='SPEC346'):
         plt.xlim(900,1600) #for SPEC346
      else:
        plt.xlim(2000,10000)

      plt.legend(loc=1);
      plt.grid(True)
      my_title='z = %.3f '%float(spec_jct.z)+'\n lines = '+spec_jct.lines
      plt.title(my_title)

      pdf_pages.savefig(fig)
      #'''
#-------------------------------------------------------------------------------------------------------
# histogramme des z_spec de fors2
def hist_zspec():
    n_sed=0
    z_spec=np.array([[]])
    for i in lst:
      n_sed=n_sed+1
      base=os.path.basename(i).split('.')[0]
      #id=base[4:-1]
      id=base[4:]
      spec=SED_jct(id)
      zspec=np.array([spec.z])
      z_spec=np.append(z_spec,zspec)
    print(('n_sed =',n_sed))
    #print(z_spec)

    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    plt.hist(z_spec, bins=400, range=[0,4.], log=False, color='r', linewidth = 3, fill=False, histtype='step', label='z_spec')
    plt.ylim(0.,60)
    plt.xlabel('z_spec')
    plt.grid(True)
    plt.legend(loc=1)
    plt.title('Fors2 spectrometric redshifts distribution\n '+'n_sed= '+str(n_sed))
    pdf_pages.savefig(fig)
#-------------------------------------------------------------------------------------------------------
# tri des spectres fors2 par couleur
def color_sort():
    l_red   =[]
    l_medium=[]
    l_blue  =[]
    
    for i in lst :
        #print('plot SED from :',i)
        #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        fig = plt.figure(figsize=(12, 6), dpi=100)
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        id_low ,=np.where(np.logical_and(low_b<spec.wave,spec.wave<4000))
        id_high,=np.where(np.logical_and(4000<spec.wave,spec.wave<high_b))
        x_low =spec.wave[id_low]
        y_low =spec.flux[id_low]*tmp
        x_high=spec.wave[id_high]
        y_high=spec.flux[id_high]*tmp
        mean_ylow =np.mean(y_low)
        mean_yhigh=np.mean(y_high)
        
        if (mean_ylow/mean_yhigh<0.44):
          tag= 'red'
          color='r-'
          l_red=np.append(l_red,i)
        elif (mean_ylow/mean_yhigh>0.91):
          tag= 'blue'
          color='b-'
          l_blue=np.append(l_blue,i)
        else :
          tag= 'medium'
          color='g-'
          l_medium=np.append(l_medium,i)

    print(('red    = ',list(set(l_red))))
    print(('len(red) = ',len(red)))
    print(('medium = ',list(set(l_medium))))
    print(('len(medium) = ',len(medium)))
    print(('blue   =  ',list(set(l_blue))))
    print(('len(blue) = ',len(blue)))

    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    for i in l_red :
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        plt.semilogy(spec.wave,spec.flux*tmp                  ,'r-', linewidth = 2)
    plt.xlabel(r'$\lambda$ [$\AA$]',size=18)
    plt.ylabel(r'Flux (arbitrary units)',size=18)
    plt.xlim(700,9000)
    plt.ylim(1e-17,2e-13)
    plt.grid(True)
    plt.legend(loc=1); #haut droit
    plt.title('red SEDs (no exctinction applied)')
    pdf_pages.savefig(fig)

    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    for i in l_medium :
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        plt.semilogy(spec.wave,spec.flux*tmp                  ,'g-', linewidth = 2)
    plt.xlabel(r'$\lambda$ [$\AA$]',size=18)
    plt.ylabel(r'Flux (arbitrary units)',size=18)
    plt.xlim(700,9000)
    plt.ylim(1e-17,2e-13)
    plt.grid(True)
    plt.legend(loc=1); #haut droit
    plt.title('medium SEDs (Prevot et al)')
    pdf_pages.savefig(fig)

    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    for i in l_blue :
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        plt.semilogy(spec.wave,spec.flux*tmp                  ,'b-', linewidth = 2)
    plt.xlabel(r'$\lambda$ [$\AA$]',size=18)
    plt.ylabel(r'Flux (arbitrary units)',size=18)
    plt.xlim(700,9000)
    plt.ylim(1e-17,2e-13)
    plt.grid(True)
    plt.legend(loc=1); #haut droit
    plt.title('blue SEDs (Calzetti et al)')
    pdf_pages.savefig(fig)


    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    for i in l_red :
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        plt.semilogy(spec.wave,spec.flux*tmp                  ,'r-', linewidth = 2)
    for i in l_medium :
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        plt.semilogy(spec.wave,spec.flux*tmp                  ,'g-', linewidth = 2)
    for i in l_blue :
        name_file=i+ext+'_BC.txt' # BC spectra sans extinction
        spec=SED(path_BC+name_file)
        tmp=norm/spec.get_scale(bounds=(b1,b2))
        plt.semilogy(spec.wave,spec.flux*tmp                  ,'b-', linewidth = 2)
    plt.xlabel(r'$\lambda$ [$\AA$]',size=18)
    plt.ylabel(r'Flux (arbitrary units)',size=18)
    plt.xlim(700,9000)
    plt.ylim(1e-17,2e-13)
    plt.grid(True)
    plt.legend(loc=1); #haut droit
    pdf_pages.savefig(fig)
    
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# histogramme des variances des differences entres spectres raw et spectres SL
def hist_diff_raw_sl():
    var_diff=[]
    for i in lst :
      #get raw spectra for first identical SED
      filename=path_out_jct+i+'.txt'
      if (os.path.exists(filename) != True):
         print(filename)
      #'''
      spec=SED(filename)
      x=spec.wave
      if ((min(x)<3900)and(max(x)>3950)):
        y=spec.flux*norm/spec.get_scale(bounds=(3900,3950))
      else:
        y=spec.flux*norm/abs(spec.get_scale(bounds=(min(x)+100,min(x)+150)))

      #get EXTINCTED spectra for first identical SED
      name_file=i+ext+'_BC_ext.txt' # BC spectra AVEC extinction par population
      spec_SL_ext=SED(path_BC_ext+name_file)
      id,=np.where(np.logical_and(min(x)<spec_SL_ext.wave,spec_SL_ext.wave<max(x)))
      x_SL_ext=spec_SL_ext.wave[id]
      y_SL_ext=spec_SL_ext.flux[id]*norm/spec_SL_ext.get_scale(bounds=(3900,3950))
      y_SL_ext_interp=np.interp(x,x_SL_ext,y_SL_ext)
      #diff=(y-y_SL_ext_interp)/y_SL_ext_interp
      #diff=(y-y_SL_ext_interp)
      diff=(y-y_SL_ext_interp)/(y+y_SL_ext_interp)*2.
      #var=np.sqrt(sum( ((y-y_SL_ext_interp)/(y+y_SL_ext_interp))**2 )/len(y) )
      var =np.var(diff)

      var_diff=np.append(var_diff,var)
    print((min(var_diff),max(var_diff)))

    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    plt.hist(var_diff, bins=np.logspace(-4, 3.0, 100), log=False, color='r', linewidth = 3, fill=False, histtype='step', label='var_diff')
    plt.gca().set_xscale("log")
    #plt.ylim(0.,35)
    plt.xlabel('var_diff')
    plt.grid(True)
    plt.legend(loc=1)
    plt.title('Variance distribution of raw and SL BC_full_ext seds difference')
    pdf_pages.savefig(fig)
      
#-------------------------------------------------------------------------------------------------------
# recherche des spectres qso:
def qso():
    print('Looking for QSO/OIII lines')
    for i in lst:
      base=os.path.basename(i).split('.')[0]
      id=base[4:]
      spec=SED_jct(id)
      lines=spec.lines.split(',')
      #print('lines =',i,lines)
      #if ('[OIII]' in lines):
      #if ('H{alpha}' in lines):
      if ('(QSO)' in lines):
        print((i, lines))
#-------------------------------------------------------------------------------------------------------
# tri des z_spec
def count_zspec():
    prt_count=0
    z_spec=[]
    for i in lst:
      base=os.path.basename(i).split('.')[0]
      id=base[4:]
      spec=SED_jct(id)
      zspec=spec.z
      z_spec=np.append(z_spec,'%.5f'%float(zspec)) #liste des z_spec 
    for j in z_spec:
      n_occ=np.count_nonzero(z_spec == j)
      if (n_occ>=3):
         id,=np.where(z_spec==j)
         print(id)
         print((j,n_occ))
         for k in id:
           print((lst[int(k)]))
#-------------------------------------------------------------------------------------------------------
def fors2_jct_SL():
  lst=[]
  list=glob.glob(path_raw_jct+'*.txt')
  for f in list:
    base=os.path.basename(f).split('.')[0]
    id=base[4:-1]
    #print(id)
    spec=SED_jct(id)
    if (float(spec.z)!=-1) :
      lst=np.append(lst,'SPEC'+id)
      file_out=path_out_jct+'SPEC'+str(id)+'.txt'
      h=open(file_out,'w')
      x=spec.wave
      y=spec.flux
      for i in range(len(x)):
        h.write("%f %f\n"%(float(x[i]),float(y[i])))
      h.close()

  return lst  
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# recherche de spectres identiques
def compare_spectra(plot_out): #now 
    lst_tmp=lst[:]
    z_spec=[]
    ra=[]
    dec=[]
    print('reading spectra:')
    #----------------------------------
    #building list of normalized fluxes (lst_y):

    # Filling the lists with the first SED:
    i_tmp=lst_tmp[0]
    # SL spectra : --------------------
    #get spectra with NO extinction :
    name_file=i_tmp+ext+'_BC.txt' # BC spectra sans extinction
    spec=SED(path_BC+name_file)
    id ,=np.where(np.logical_and(low_b<spec.wave,spec.wave<high_b))
    x =spec.wave[id]
    tmp_SL=norm/spec.get_scale(bounds=((low_b+high_b)/2.-delta,(low_b+high_b)/2.+delta))
    lst_y =np.array([spec.flux[id]*tmp_SL])
    lst_tmp.remove(i_tmp)
    #----------------------------------
    # raw spectra : -------------------
    #get z,ra,dec info for this spectra:
    base=os.path.basename(i_tmp).split('.')[0]
    id=base[4:]
    spec2=SED_jct(id)
    z_spec=np.append(z_spec,spec2.z)
    ra =np.append(ra ,spec2.ra)
    dec=np.append(dec,spec2.dec)
    #----------------------------------
    #----------------------------------
    # Filling the lists for all SEDs:
    for i in lst_tmp :
      # SL spectra : --------------------
      name_file=i+ext+'_BC.txt' # BC spectra sans extinction
      spec=SED(path_BC+name_file)
      id ,=np.where(np.logical_and(low_b<spec.wave,spec.wave<high_b))
      tmp_SL=norm/spec.get_scale(bounds=((low_b+high_b)/2.-delta,(low_b+high_b)/2.+delta))
      y =np.array([spec.flux[id]*tmp_SL])
      lst_y=np.append(lst_y,y,axis=0)
      # raw spectra : -------------------
      #get z,ra,dec info for this spectra:
      base=os.path.basename(i).split('.')[0]
      id=base[4:]
      spec2=SED_jct(id)
      z_spec=np.append(z_spec,spec2.z)
      ra =np.append(ra ,spec2.ra)
      dec=np.append(dec,spec2.dec)
    #----------------------------------
    print('search for identical spectra')
    same_spec=[]
    same_spec_cleaned=[]
    unique_spec=[]
    rejected_spec=[]
    i_lst=list(range(len(lst_y)))
    j_lst=i_lst[:]
    for i in i_lst:
      #print('check 1 : ',i,lst[i])
      j_lst.remove(i)
      #print('my test :',lst[i])
      for j in j_lst:
        #print('check 2 : ',j,lst[j])
        #diff=(lst_y[i]-lst_y[j])/lst_y[i] #this was a bug as it depends from the i in the denominator ! 
        diff=(lst_y[i]-lst_y[j])/(lst_y[i]+lst_y[j])*2.
        mean = np.mean(diff)
        var  = np.var(diff)
        if ((var<var_cut) and (i!=j)): # check that varience of the difference of two SL extrapoladion is not too high
          # plot of the two SL (no extinction) spectra:
          if plot_out : plot_fig(x,lst_y[i],'k-',lst[i]+' SL',x,lst_y[j],'c-',lst[j]+' SL',title='')
          
          #get raw spectra for first identical SED:
          filename1=path_out_jct+lst[i]+'.txt'
          spec1=SED(filename1)
          x1=spec1.wave
          z_spec1=z_spec[i]
          #get raw spectra for second identical SED:
          filename2=path_out_jct+lst[j]+'.txt'
          spec2=SED(filename2)
          x2=spec2.wave
          z_spec2=z_spec[j]
          if ((min(x1)<min(x2) and max(x1)<min(x2)) or (min(x2)<min(x1) and max(x2)<min(x1))) :
             #print(' cas 0 #####################################')
             b_min_1= min(x1)
             b_max_1= max(x1)
             name_file=lst[i]+ext+'_BC_ext.txt' # BC spectra AVEC extinction par population
             spec1_SL_ext=SED(path_BC_ext+name_file)
             norm_1 = spec1_SL_ext.get_scale(bounds=((b_min_1+b_max_1)/2.-delta,(b_min_1+b_max_1)/2.+delta))
             tmp1_raw=norm_1/spec1.get_scale(bounds=((b_min_1+b_max_1)/2.-delta,(b_min_1+b_max_1)/2.+delta))

             b_min_2= min(x2)
             b_max_2= max(x2)
             name_file=lst[j]+ext+'_BC_ext.txt' # BC spectra AVEC extinction par population
             spec2_SL_ext=SED(path_BC_ext+name_file)
             norm_2  =spec2_SL_ext.get_scale(bounds=((b_min_2+b_max_2)/2.-delta,(b_min_2+b_max_2)/2.+delta))
             tmp2_raw=norm_2/spec2.get_scale(bounds=((b_min_2+b_max_2)/2.-delta,(b_min_2+b_max_2)/2.+delta))
          elif ((min(x1)<min(x2)) and (max(x1)>min(x2)) and (max(x1)<max(x2))):
             #print(' cas 1 #####################################')
             b_min_1= min(x2)
             b_max_1= max(x1)
             b_min_2= b_min_1 
             b_max_2= b_max_1
          elif ((min(x2)<min(x1)) and (max(x2)>min(x1)) and (max(x2)<max(x1))):
             #print(' cas 2 #####################################')
             b_min_1= min(x1)
             b_max_1= max(x2)
             b_min_2= b_min_1 
             b_max_2= b_max_1
          elif ((min(x2)<min(x1)) and (max(x1)<max(x2))):
             #print(' cas 3 #####################################')
             b_min_1= min(x1)
             b_max_1= max(x1)
             b_min_2= b_min_1 
             b_max_2= b_max_1
          elif ((min(x1)<min(x2)) and (max(x2)<max(x1))):
             #print(' cas 4 #####################################')
             b_min_2= min(x2)
             b_max_2= max(x2)
             b_min_1= b_min_2 
             b_max_1= b_max_2

          tmp1_raw=norm/spec1.get_scale(bounds=((b_min_1+b_max_1)/2.-delta,(b_min_1+b_max_1)/2.+delta))
          tmp2_raw=norm/spec2.get_scale(bounds=((b_min_2+b_max_2)/2.-delta,(b_min_2+b_max_2)/2.+delta))
          y1=spec1.flux*tmp1_raw
          y2=spec2.flux*tmp2_raw

          #get EXTINCTED spectra for first identical SED
          name_file=lst[i]+ext+'_BC_ext.txt' # BC spectra AVEC extinction par population
          spec1_SL_ext=SED(path_BC_ext+name_file)
          id1,=np.where(np.logical_and(min(x1)<spec1_SL_ext.wave,spec1_SL_ext.wave<max(x1)))
          x1_SL_ext=spec1_SL_ext.wave[id1]
          tmp1_SL_ext=norm/spec1_SL_ext.get_scale(bounds=((b_min_1+b_max_1)/2.-delta,(b_min_1+b_max_1)/2.+delta))
          y1_SL_ext=spec1_SL_ext.flux[id1]*tmp1_SL_ext
          y1_SL_ext_interp=np.interp(x1,x1_SL_ext,y1_SL_ext)
          diff1=(y1-y1_SL_ext_interp)/y1_SL_ext_interp
          var1=np.var(diff1)

          #get EXTINCTED spectra for second identical SED
          name_file=lst[j]+ext+'_BC_ext.txt' # BC spectra AVEC extinction par population
          spec2_SL_ext=SED(path_BC_ext+name_file)
          id2,=np.where(np.logical_and(min(x2)<spec2_SL_ext.wave,spec2_SL_ext.wave<max(x2)))
          x2_SL_ext=spec2_SL_ext.wave[id2]
          tmp2_SL_ext=norm/spec2_SL_ext.get_scale(bounds=((b_min_2+b_max_2)/2.-delta,(b_min_2+b_max_2)/2.+delta))
          y2_SL_ext=spec2_SL_ext.flux[id2]*tmp2_SL_ext
          y2_SL_ext_interp=np.interp(x2,x2_SL_ext,y2_SL_ext)
          diff2=(y2-y2_SL_ext_interp)/y2_SL_ext_interp
          var2=np.var(diff2)
          #plot of raw identical spectra:
          if plot_out :
             plot_fig(x1,y1,'k-',lst[i]+' raw',x2,y2,'c-',lst[j]+' raw',title='')
             my_title='z = %.3f ; var = %.1e'%(z_spec1,var1)
             plot_fig(x1,y1,'m-',lst[i]+' raw',x1_SL_ext,y1_SL_ext,'y-',lst[i]+' SL BC_ext_full',title=my_title)
             my_title='z = %.3f ; var = %.1e'%(z_spec2,var2)
             plot_fig(x2,y2,'m-',lst[j]+' raw',x2_SL_ext,y2_SL_ext,'y-',lst[j]+' SL BC_ext_full',title=my_title)

          #print('\nsimilar spectra: ')
          print('\n')
          print((lst[i],lst[j]))
          print(('z1, z2, z1/z2 :',z_spec1,z_spec2,z_spec1/z_spec2))
          print(('ra1 ,ra2      : ', ra[i], ra[j]))
          print(('dec1,dec2     : ',dec[i],dec[j]))
          print(('var1,var2     : ',var1,var2))
          same_spec=np.append(same_spec,lst[i])
          same_spec=np.append(same_spec,lst[j])
          if (var1<var2):
            if (lst[i] in rejected_spec)==False:
              same_spec_cleaned=np.append(same_spec_cleaned,lst[i])
              print(('keeping ',lst[i]))
              rejected_spec=np.append(rejected_spec,lst[j])
              same_spec_cleaned=list(set(same_spec_cleaned))
              if lst[j] in same_spec_cleaned: same_spec_cleaned.remove(lst[j])  
          else:
            if (lst[j] in rejected_spec)==False:
              same_spec_cleaned=np.append(same_spec_cleaned,lst[j])
              rejected_spec=np.append(rejected_spec,lst[i])
              print(('keeping ',lst[j]))
              same_spec_cleaned=list(set(same_spec_cleaned))
              if lst[i] in same_spec_cleaned: same_spec_cleaned.remove(lst[i])
    unique_spec=lst[:]
    for i in list(set(same_spec)):
        unique_spec.remove(i)
    final_cleaned_list=list(set(np.append(same_spec_cleaned,unique_spec)))

    print('\n')
    print(('lst                = ',lst,'\n'))
    print(('same_spec          = ',same_spec,'\n'))
    print(('same_spec_cleanded = ',same_spec_cleaned,'\n'))
    print(('unique_spec        = ',unique_spec,'\n'))
    print(('len (lst, same_spec, same_spec_cleaned,unique_spec) :',len(lst),len(same_spec),len(same_spec_cleaned),len(unique_spec),'\n'))
    print(('final_cleaned_list = ',final_cleaned_list,'\n'))
    print((len(final_cleaned_list),'\n'))
#-------------------------------------------------------------------------------------------------------
def hist_var() :
    lst_tmp=lst[:]
    var_ij=[]
    print('reading spectra:')
    #----------------------------------
    #building list of normalized fluxes (lst_y):

    # Filling the lists with the first SED:
    i_tmp=lst_tmp[0]
    # SL spectra : --------------------
    #get spectra with NO extinction :
    name_file=i_tmp+ext+'_BC.txt' # BC spectra sans extinction
    spec=SED(path_BC+name_file)
    id ,=np.where(np.logical_and(low_b<spec.wave,spec.wave<high_b))
    x =spec.wave[id]
    tmp_SL=norm/spec.get_scale(bounds=((low_b+high_b)/2.-delta,(low_b+high_b)/2.+delta))
    lst_y =np.array([spec.flux[id]*tmp_SL])
    lst_tmp.remove(i_tmp)
    #----------------------------------
    # raw spectra : -------------------
    base=os.path.basename(i_tmp).split('.')[0]
    id=base[4:]
    spec2=SED_jct(id)
    #----------------------------------
    #----------------------------------
    # Filling the lists for all SEDs:
    for i in lst_tmp :
      # SL spectra : --------------------
      name_file=i+ext+'_BC.txt' # BC spectra sans extinction
      spec=SED(path_BC+name_file)
      id ,=np.where(np.logical_and(low_b<spec.wave,spec.wave<high_b))
      tmp_SL=norm/spec.get_scale(bounds=((low_b+high_b)/2.-delta,(low_b+high_b)/2.+delta))
      y =np.array([spec.flux[id]*tmp_SL])
      lst_y=np.append(lst_y,y,axis=0)

    #----------------------------------
    print('search for identical spectra')
    i_lst=list(range(len(lst_y)))
    j_lst=i_lst[:]
    for i in i_lst:
      j_lst.remove(i)
      for j in j_lst:
        diff=(lst_y[i]-lst_y[j])/(lst_y[i]+lst_y[j])*2.
        mean = np.mean(diff)
        var  = np.var(diff)
        var_ij=np.append(var_ij,var)
    var_ij=np.sort(var_ij)    
    print(('len(var_ij)  = ',len(var_ij)))

    id_cut ,=np.where(var_ij<var_cut)
    print(('len(var_cut) = ',len(id_cut)))
    
    #fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    fig = plt.figure(figsize=(12, 6), dpi=100)
    plt.hist(var_ij, bins=np.logspace(-4, 3.0, 100), log=False, color='r', linewidth = 3, fill=False, histtype='step', label='var_ij')
    plt.gca().set_xscale("log")
    plt.xlabel('var_ij')  

    plt.grid(True)
    plt.legend(loc=1)
    plt.title('Variance distribution of raw_i and raw_j  seds difference')
    pdf_pages.savefig(fig)
        
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

