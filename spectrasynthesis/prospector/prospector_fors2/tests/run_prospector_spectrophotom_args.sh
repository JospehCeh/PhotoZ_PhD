#/bin/sh -x

#conda activate prospector

while getopts i: flag
do
    case "${flag}" in
        i) id=${OPTARG};;
    esac
done


python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid $i --datamode spectrophotom \
--optimize \
--outfile=run_optimize

python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid $i --datamode spectrophotom \
--optimize --emcee \
--outfile=run_optimize_emcee


python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid $i --datamode spectrophotom \
--optimize --dynesty \
--outfile=run_optimize_dynesty
