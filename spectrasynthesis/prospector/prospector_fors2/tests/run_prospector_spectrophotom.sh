#/bin/sh -x

#conda activate prospector

python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid 214 --datamode spectrophotom \
--optimize \
--outfile=run_optimize

python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid 214 --datamode spectrophotom \
--optimize --emcee \
--outfile=run_optimize_emcee


python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid 214 --datamode spectrophotom \
--optimize --dynesty \
--outfile=run_optimize_dynesty
