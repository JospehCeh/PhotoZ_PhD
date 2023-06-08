#! /bin/sh
#
#
ggc_index=1
data="--ggc_data=../data/ggc.h5 --ggc_index=${ggc_index} --mask_elines"
opts="--jitter_model --add_realism --continuum_order=15"
fit="--dynesty --nested_method=rwalk"

python ggc.py $fit $opts $data \
              --outfile=output_examples/ggc_id$ggc_index
