#! /bin/sh

zred=0.1
igal=01
sfh="--illustris_sfh_file=../data/illustris/illustris_sfh_galaxy${igal}.dat"
mock="--logzsol=-0.3 --logmass=10 --mass=1e10 --dust2=0.5"
data="--snr_phot=0 --snr_spec=100 --add_noise"
fit="--dynesty --nested_method=rwalk"

# Non-parametric
model="--continuum_order 0  --nbins_sfh 14"
python illustris.py $fit $model $data \
                    $mock $sfh --zred=$zred \
                    --outfile=output/illustris_gal${igal}_nonpar

# parametric
model="--continuum_order 0  --parametric_sfh"
python illustris.py $fit $model $data \
                    $mock $sfh --zred=0.01 \
                    --outfile=output/illustris_gal${igal}_par
