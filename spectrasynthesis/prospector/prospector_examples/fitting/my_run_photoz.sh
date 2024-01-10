#! /bin/sh

opts="--free_igm --add_neb --complex_dust --free_neb_met"
fit="--dynesty --nested_method=rwalk"

python photoz_GNz11.py $fit $opts --nbins_sfh=5 --outfile output_examples/photoz_gnz11
