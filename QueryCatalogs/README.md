# README.md on Query

- author : Sylvie Dagoret-Campagne
- creation  date mai 2023
- last update : June 8th 2023




## subdirs essentials


- **QueryKIDS**: Search in catalogs for KIDS. Should download data from ESO web page into fits file 

- **QueryGALEX**: Search in catalogs for Galex. Should query using MAST astroquery services.

- **MergeCatalogsF2KIDSGLX**:  Merge extracted matched catalogs Galex and KIDS 




## subdirs with notebooks testing tools

- **QueryVizierandNEDServers**: Search in CDS : Simbad, Vizier and NES with astroquery
      
- **QueryIR**: Search in InfraRed missions,

- **QueryMAST**: Search in MAST services                        

- **QueryKIDS**: Search in catalogs for KIDS                         

- **QueryCDSMocServer**: Query skymap footprint of surveys, buggy !     

- **QueryESO**:    Test Queries at ESO (need to go on ESO web server with a registered account)


## top notebooks for testing astroqueries

- **SimbadOnRXCluster.ipynb** : simple notebook to probe what *Simbad/CDS* servers know about the RXJ0054.0-2823 galaxy cluster

- **Fors2SL_querycatalogs.ipynb** : more elaborated notebook to query info on Fors2 catalog. For this the For2 spectra file in observation frame is open



## Data container

-  *./data* : Input and Output from/for above notebooks:
