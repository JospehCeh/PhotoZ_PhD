# README.md

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : 2022-12-19
- last update : 2023-05-23
- purpose :Study Fors2


## ViewStandardSED.ipynb
- creation 2022/12/22
- view spectra of other datasets

## ExploreFors2.ipynb (DEPRECATED)
- first exploration of Fors2 spectra
- mandatory to run this notebook first in order to create seds file in ./fors2out/seds (or use short version below)
- don't forget to create the output directory StudyFors2SED/fors2out/seds before running

## ExploreFors2_short.ipynb (MUST RUN)
- similar to ExploreFors2.ipynb, but using python script as the library
- fitst exploration of Fors2 spectra with python script import
- mandatory to run this notebook first in order to create seds file in ./fors2out/seds of used version above
- don't forget to create the output directory StudyFors2SED/fors2out/seds before running

## ExploreFors2inRestFrame.ipynb (DEPRECATED)
- build an astropy table
- Does nothing more

## ExploreFors2_comparespectra.ipynb
- creation 2022/12/22
- Compare the spectra one by one, all together

## ExploreSL_comparespectra.ipynb
- creation 2022/12/23
- Compare the spectra one by one

##  ExploreFors2_viewspectra1by1.ipynb
- view spectra one by one , including emission lines
- creation 2022/12/23


## ExploreFors2_viewspectra1by1_CompareSL.ipynb
- Compare SL spectra with FORS2 spectra redshifted at z=0


## ExploreFors2_viewspectra1by1_CompareSL_t.ipynb
- Compare SL spectra with FORS2 spectra redshifted at z=0



# preparation of spectrum synthesis

## ExploreFors2inOriginalFrame.ipynb (MUST RUN)
- prepare to use restframe unredshifted spectra (original spectra) and extract raw spectra inside a local dir ./raw

## ExploreFors2_viewspectra1by1_raw_sdc.ipynb
- View restframe unrestshifted spectra (original spectra)

## convertSLspectratohdf5.ipynb
- Convert SL spectra in h2 file (same in DeepLearning dir )


## convertFors2spectratohdf5.ipynb (Under dev)
- Convert Fors2 (original unredshifted spectra) in h2 file 

