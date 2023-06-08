# README.md

- author Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : 2023-06-08
- last update : 2023-06-08


The goal is to find the galaxies in KIDS catalog the FOV of cluster RXJ0054.0-2823 .
Apparently no catalog can be obtained from astroquery (QueryFors2InKIDSCatalogs.ipynb).
We better to to ESO data page for KIDS at https://www.eso.org/qi/catalogQuery/index/260 https://kids.strw.leidenuniv.nl/DR4/access.php. 
I have downloaed the catalog into a fits file.
Thus one must use  **QueryFors2InKIDSCatalogsFits.ipynb** or **ViewFors2InKIDSCatalogsFits.ipynb**


## notebooks


- *QueryFors2InKIDSCatalogs.ipynb* : Query inside astroquery : does not wok, thus deprecated     
- *QueryFors2InKIDSCatalogsFits.ipynb* : Simple match Fors2/Kids catalog.  Output file info_fors2Kidscrossmatch.csv.
- *ViewFors2InKIDSCatalogsFits.ipynb*  : Simple match Fors2/Kids catalog and perform additional check by plotting skymap of the match. Output file info_fors2Kidscrossmatch.csv. This is the must use notebook. 

## output file

- info_fors2Kidscrossmatch.csv


