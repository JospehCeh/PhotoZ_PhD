#/bin/sh -x

#conda activate prospector

ID=411

echo "proceed SPEC ${ID} for spectro"

python fit_params_fors2_v2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid ${ID} --datamode spectro \
--optimize \
--outfile=run_optimize

python fit_params_fors2_v2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid ${ID} --datamode spectro \
--optimize --emcee \
--outfile=run_optimize_emcee


python fit_params_fors2_v2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid ${ID} --datamode spectro \
--optimize --dynesty \
--outfile=run_optimize_dynesty
