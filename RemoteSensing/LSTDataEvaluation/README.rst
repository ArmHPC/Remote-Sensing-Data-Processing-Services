LST Data Evaluation Service using RemoteSensing and Observations data
================================


Input Data
================================

The input data required by the LST Data Evaluation Service are the following:

1. Station Coordinates: This data should be in a CSV file format and should include the following columns:

- Station ID: A unique identifier for each station
- Latitude: The latitude of the station in decimal degrees
- Longitude: The longitude of the station in decimal degrees

2. TIF Files: These are satellite images in TIF file format containing the following bands:

- For Landsat: Tar file containing Band 10: Thermal Infrared (TIR) band, Band 11: TIR band, Band 4: Near Infrared (NIR) band, Band 3: Red band
- For VIIRS: TIF files that containes already calculatied LST for the region

3. Observations: This data should be in a CSV file format and should include the following columns:

- Station ID: The name of the station
- Date: The date of the observation in mm/dd/yyyy format
- Time: The time of the observation in hh:mm:ss format
- Temperature: The temperature observed at the station in Kelvin

Output Data
================================

The LST Data Evaluation Service provides the following output data:

- LST data evaluation results for each station in the CSV file format


Steps to use LST Data Evaluation Service
================================

Step: Execute extract_lst.py to calculate estimated LST from Landsat images using the stations.csv data

Step 1: Prepare input data
Prepare the input data in the CSV file format for station coordinates and TIF file format for satellite images.

Step 2: Upload input data
Upload the input data to the LST Data Evaluation Service. You can upload the data by clicking on the "Upload" button on the service interface and selecting the CSV file and TIF files.




.. Step 3: Set Parameters
.. Set the parameters for LST data evaluation. These parameters include:

.. Satellite Data Type: Select the satellite data type you want to evaluate (Landsat or VIIRS).
.. Atmospheric Correction Method: Select the atmospheric correction method you want to use for the evaluation.
.. LST Calculation Method: Select the LST calculation method you want to use for the evaluation.
.. Spatial Resolution: Select the spatial resolution of the output data.
.. Time Range: Select the time range of the satellite images.
.. Step 4: Run the Service
.. Click on the "Run" button to start the LST data evaluation process.

.. Step 5: Download Output
.. Once the LST data evaluation process is complete, you can download the output data in CSV file format.


Conclusion
================================

The LST Data Evaluation Service is a useful tool for evaluating LST data using Landsat or VIIRS 
satellite images and station coordinates. By following the steps mentioned above, 
you can easily use the service to evaluate LST data for different station coordinates and satellite images.