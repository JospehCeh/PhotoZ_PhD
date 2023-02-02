#!/bin/zsh

export PYTHONPATH=$HOME/setmeup
#export PYTHONPATH=$HOME/labo/lsst/photoz/dc1/qq_plot

RACINE=$HOME/00_labo/lsst/photoz/lephare
export LEPHAREDIR=${RACINE}/lephare_dev
export LEPHAREWORK=${RACINE}/output
#cd $LEPHAREDIR/test;

export EXT_LAW='HZ4' #prevot
#export EXT_LAW='HZ5' #Calzetti

#export ANA_TYPE='cww'
#export ANA_TYPE='brown'
#export ANA_TYPE='brown_illustrative'

#export ANA_TYPE='fors2' #useless
#export ANA_TYPE='brown_rebuild'
#export ANA_TYPE='brown_wide'

export ANA_TYPE='fors2_raw'
#export ANA_TYPE='fors2_raw_eg' #useless
#export ANA_TYPE='fors2_test' #useless

export RUN_TYPE='full'
#export RUN_TYPE='illustrative'
#export RUN_TYPE='test'

#export PLOT_TYPE='detailed'
export PLOT_TYPE=''

#for starlight :
export BASE_TAG='BC03N' #45  stars
#export BASE_TAG='BC03S' #150 stars
#export BASE_TAG='JM'

export CONFIG_TAG='conf1'
#export CONFIG_TAG='conf2' #better_fit

