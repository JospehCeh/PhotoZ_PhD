# README.md

- creation : 2023/04/23
- update : 2023/04/26


## Environnment Fidle + rubin_sim



- Sylvie Dagoret-Campagne

## Train VAE and save model
- notebook file : **astroml_VAE.ipynb**           
- input file : *SLspectra.hdf5* 

## Read saved model
- notebook file : **astroml_VAE_readmodel.ipynb**

## Analyse the model
- *astroml_VAE_readmodel_andstudy.ipynb*

## Draw the model

   tensorboard --logdir=./


- Working with tensorboard : *astroml_VAE_train_and_drawmodel.ipynb*    
- Not working with tensorboard : *astroml_VAE_readmodel_anddrawmodel.ipynb*


## More developped analysis with 1 or 2 latent varibles

- *astroml_VAE_trainmodel1latentvariable.ipynb*
- *astroml_VAE_trainmodel2latentvariables.ipynb*


## With rubin : compare color-color plot with latent space variable

- *astroml_VAE_readmodel_anrubinsim.ipynb*
