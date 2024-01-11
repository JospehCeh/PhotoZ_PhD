#! /bin/sh
#
data="--objname 92942 --zred=0.073"
model="--continuum_order=12 --add_neb --free_neb_met --marginalize_neb"
model=$model" --nbins_sfh=8 --jitter_model --mixture_model"
fit="--dynesty --nested_method=rwalk --nlive_batch=200 --nlive_init 500"
fit=$fit" --nested_dlogz_init=0.01 --nested_posterior_thresh=0.03"

python psb_params.py $fit $model $data \
                     --outfile=output_examples/psb_92942
