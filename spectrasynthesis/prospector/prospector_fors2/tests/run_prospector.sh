#/bin/sh -x

conda activate prospector

python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid 214 --datamode photom


