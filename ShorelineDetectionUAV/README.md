# Scripts for Datacube environment

Description: This Python notebook allows users to monitor small water area and particularly study of changes of shorelines combining Sentinel 2 and UAV photogrammetry. The script is structured as follows:

Step 1: Preprocessing

Read RGB image from UAV image using bands and take from metadata the boundaries.
Using that boundaries, fetch data from Datacube for Sentinel 2 for the closest date to the UAV image data. The used bands are ['green','red', ‘blue’, 'nir', 'swir1', 'swir2' ].

Step 2: To identify and extract water bodies

BandRatio = B3/B8
McFeeters = (B3 - B8) / (B3 + B8)
MSAVI12 = (2B8 + 1 - xu.sqrt((2B8)(2B8) - 8*(B8 - B4))) / 2
MNDWI1 = (B3 - B11) / (B3 + B11)
MNDWI2 = (B3 - B12) / (B3 + B12)

Step 3: To extract coastline

K-means clustering method
Gaussian Blur
Canny edge detection
After these steps we will have the shoreline matrix where in the positions of the line will have values 1 and the other positions 0.

Step 4: To compere UAV and Satelite images

Mask the shoreline matrix with initial images
Get the coordinates in (lat,long) pairs
For each Satellite image shoreline point to find the nearest points from the UAV image with Euclidean distance and calculate the RMSE for the whole line

Script is written for running in [Jupyter Notebook](https://jupyter.org/) environment by using [Datacube API](https://datacube-core.readthedocs.io/).

