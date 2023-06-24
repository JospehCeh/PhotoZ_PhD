#/bin/sh -x

#conda activate prospector

while getopts i: flag
do
    case "${flag}" in
        i) id=${OPTARG};;
    esac
done

echo "will proceed spec ${id}"

python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid ${id} --datamode photom \
--optimize \
--outfile=run_optimize

python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid ${id} --datamode photom \
--optimize --emcee \
--outfile=run_optimize_emcee


python fit_params_fors2.py --inputfile ../../../../QueryCatalogs/data/FORS2spectraGalexKidsPhotom.hdf5 --objid ${id} --datamode photom \
--optimize --dynesty \
--outfile=run_optimize_dynesty
