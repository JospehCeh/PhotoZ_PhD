# README.md

- author Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- creation date : 2023-06-08
- last update : 2023-06-08


The goal is to find the galaxies in GALEX catalog the FOV of cluster RXJ0054.0-2823 .
I tried to download data from web page https://galex.stsci.edu/GR6/?page=mastform . But apparently the selection radius is wrong.
A more complete information is obtained via the MAST server in astroquery.
Thus one must use **ViewFors2andGALEXCatalogMAST.ipynb**


## Notebooks

- **QueryFors2OnGALEXCatalogCSV.ipynb** :  Simple match Fors2/Kids catalog           
- **ViewFors2andGALEXCatalogCSV.ipynb** :  Simple match Fors2/Kids catalog  but it shows incomplete coverage of FORS2 FOV should not use      


- **ViewFors2andGALEXCatalogMAST.ipynb** : Simple match Fors2/Kids catalog and perform additional check by plotting skymap of the match. Output file **info_fors2GALEX_frommast_crossmatch.csv**. This is the must use notebook. 

## Output

- **info_fors2GALEX_fromweb_crossmatch.csv** : Incomplete, do not use
- **info_fors2GALEX_frommast_crossmatch.csv**: Complete output from MAST , a must use result.