#! /bin/sh


mock="--zred=0.1 --tau=4 --tage=12 --logzsol=-0.3 --mass=1e10 --dust2=0.3"
opts="--add_duste --add_neb"
data="--add_noise --mask_elines --continuum_optimize"
fit="--dynesty --nested_method=rwalk"

# photometry only
python specphot_demo.py $fit $mock $opts --zred_disp=1e-3 $data --snr_spec=0 --snr_phot=20 --outfile=output/mock_parametric_phot

# spectroscopy only
python specphot_demo.py $fit $mock $opts --zred_disp=1e-3 $data --snr_spec=10 --snr_phot=0 --outfile=output/mock_parametric_spec

# photometry + spectroscopy
python specphot_demo.py $fit $mock $opts --zred_disp=1e-3 $data --snr_spec=10 --snr_phot=20 --outfile=../output/mock_parametric_specphot

