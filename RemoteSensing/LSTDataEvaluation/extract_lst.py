
import sys
import os
import getopt
import tarfile
import rasterio
from rasterio.plot import show
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon 

from pyproj import Transformer

import datetime

# Dir to LST folders
LANDSAT_DIR = '/Users/haykgrigoryan/Documents/IIAP/ArmeniaLSTTars'
STATIONS_DIR = 'stations.csv'
TMP_DIR = 'tmp'

def calcLSTV2(LandsatToa, TIR, K1, K2):
    ML_10 = 0.0003342
    AL_10 = 0.1
    TOA_L = ML_10 * TIR + AL_10
    
    K1_10 = K1
    K2_10 = K2
    BT = (K2_10 / np.log(K1_10 / TOA_L + 1))
    
    NDVI = (LandsatToa["b5_masked"] - LandsatToa["b4_masked"]) / (LandsatToa["b5_masked"] + LandsatToa["b4_masked"])
    
    minNdvi = np.nanmin(NDVI)
    maxNdvi = np.nanmax(NDVI)
    
    Pv = ((NDVI - minNdvi) / (maxNdvi - minNdvi)) * ((NDVI - minNdvi) / (maxNdvi - minNdvi))

    ε = 0.004 * Pv + 0.986

    return (BT / (1 + (0.00115 * BT / 1.4388) * np.log(ε)))

class Station:
    def __init__(self,station_name, latitude, longitude) -> None:
        self.station_name = station_name;
        self.latitude = latitude
        self.longitude = longitude
        self.dates = []
        self.values = []

class LandsatMetadata:
    
    def __init__(self) -> None:
        pass
        
    def setDate(self,DATE_ACQUIRED, SCENE_CENTER_TIME) -> None:
        self.DATE_ACQUIRED = DATE_ACQUIRED
        self.SCENE_CENTER_TIME = SCENE_CENTER_TIME
    
    def setBoundary(self,UL_LAT, UL_LON, UR_LAT, UR_LON, LL_LAT, LL_LON, LR_LAT, LR_LON):
        self.UL_LAT = UL_LAT
        self.UL_LON = UL_LON
        
        self.UR_LAT = UR_LAT
        self.UR_LON = UR_LON
        
        self.LL_LAT = LL_LAT
        self.LL_LON = LL_LON
        
        self.LR_LAT = LR_LAT
        self.LR_LON = LR_LON
        
        polygonArr = [(UL_LAT, UL_LON), (UR_LAT, UR_LON), (LR_LAT, LR_LON), (LL_LAT, LL_LON)]
        
        line = LineString(polygonArr)
        self.polygon = Polygon(line)
        
    def setKConstants(self, K1_10, K2_10, K1_11,K2_11):
        self.K1_10 = K1_10
        self.K2_10 = K2_10
        self.K1_11 = K1_11
        self.K2_11 = K2_11
    
    def setBands(self, B1Path,B2Path,B3Path,B4Path,B5Path,B6Path,B7Path,B8Path,B9Path,B10Path,B11Path):
        self.B1Path = B1Path
        self.B2Path = B2Path
        self.B3Path = B3Path
        self.B4Path = B4Path
        self.B5Path = B5Path
        self.B6Path = B6Path
        self.B7Path = B7Path
        self.B8Path = B8Path
        self.B9Path = B9Path
        self.B10Path = B10Path
        self.B11Path = B11Path
        
    def isInBounds(self, lat, lon) -> bool:
        return self.polygon.contains(Point(lat, lon))
    
    def getRowCol(self, bandPath, lat, lon):
        B1TIFF = rasterio.open(bandPath)
        outProj = pyproj.Proj(init='epsg:32636')
        inProj = pyproj.Proj(init='epsg:4326')
        x,y = pyproj.transform(inProj,outProj,lon,lat)
        row, col = B1TIFF.index(x, y)
        return row, col
    
    def getValueForBand(self, bandPath, lat, lon) -> int:
        B1TIFF = rasterio.open(bandPath)
        outProj = pyproj.Proj(init='epsg:32636')
        inProj = pyproj.Proj(init='epsg:4326')
        x,y = pyproj.transform(inProj,outProj,lon,lat)
        z = B1TIFF.read(1)
        row, col = B1TIFF.index(x, y)
        return z[row][col]
    
    def calcLST(self):
        B4TIFF = rasterio.open(self.B4Path)
        B5TIFF = rasterio.open(self.B5Path)
        B10TIFF = rasterio.open(self.B10Path)
        B11TIFF = rasterio.open(self.B11Path)
        ForLST = {}
        ForLST["b4"] = B4TIFF.read(1)
        ForLST['b4_masked'] = np.ma.masked_array(ForLST['b4'], mask=(ForLST['b4'] == 0))
        ForLST["b5"] = B5TIFF.read(1)
        ForLST['b5_masked'] = np.ma.masked_array(ForLST['b5'], mask=(ForLST['b5'] == 0))
        ForLST["b10"] = B10TIFF.read(1)
        ForLST['b10_masked'] = np.ma.masked_array(ForLST['b10'], mask=(ForLST['b10'] == 0))
        ForLST["b11"] = B11TIFF.read(1)
        ForLST['b11_masked'] = np.ma.masked_array(ForLST['b11'], mask=(ForLST['b11'] == 0))
        self.LSTV1 = calcLSTV2(ForLST,ForLST['b10_masked'], self.K1_10, self.K2_10)
        self.LSTV2 = calcLSTV2(ForLST,ForLST['b11_masked'], self.K1_11, self.K2_11)
    
    def read_lst_from_img(self, Img_Data, x, y):
        try:
            Y, X = Img_Data.shape
            if y < 0 or y >= Y : return float('nan')
            if x < 0 or x >= X : return float('nan')
            temp_kelvin = float(Img_Data[y, x]).__round__(1)
            return temp_kelvin if temp_kelvin > 0 else float('nan')
        except:
            return float('nan')
    
    
    def getPointVal(self, lat, lon) -> None:
        Img_Meta = rasterio.open(self.B1Path)
        transformer = Transformer.from_crs("EPSG:4326", Img_Meta.crs)
        xx, yy = transformer.transform([lat], [lon])
        pix_coords = [~Img_Meta.transform * (x, y) for x,y in zip(xx, yy)]
        pix_coords = [(int(x + 0.5), int(y + 0.5)) for x,y in pix_coords]
        
        return [self.read_lst_from_img(self.LSTV1, x, y) for x,y in pix_coords][0], [self.read_lst_from_img(self.LSTV2, x, y) for x,y in pix_coords][0]
        
    def print(self) ->None:
        print(self.DATE_ACQUIRED, self.SCENE_CENTER_TIME)
    
    

def getTarFiles(dir, tarFiles):
    filesInDir = os.listdir(dir)
    for file in filesInDir:
        filePath = dir + "/" + file
        if os.path.isdir(filePath):
            getTarFiles(filePath,tarFiles)
        else:
            if file.endswith('.tar'):
                tarFiles.append(filePath)
                
def extractTar(inFile, outDir):
    my_tar = tarfile.open(inFile)
    my_tar.extractall(outDir) # specify which folder to extract to
    my_tar.close()
    

def getTifForImage(tmp,fileName) -> LandsatMetadata:
    B1 = tmp + "/" + fileName + "_B1.TIF"
    B2 = tmp + "/" + fileName + "_B2.TIF"
    B3 = tmp + "/" + fileName + "_B3.TIF"
    B4 = tmp + "/" + fileName + "_B4.TIF"
    B5 = tmp + "/" + fileName + "_B5.TIF"
    B6 = tmp + "/" + fileName + "_B6.TIF"
    B7 = tmp + "/" + fileName + "_B7.TIF"
    B8 = tmp + "/" + fileName + "_B8.TIF"
    B9 = tmp + "/" + fileName + "_B9.TIF"
    B10 = tmp + "/" + fileName + "_B10.TIF"
    B11= tmp + "/" + fileName + "_B11.TIF"
    MTL= tmp + "/" + fileName + "_MTL.xml"
    
    B1TIFF = rasterio.open(B1)
    
    metadata = LandsatMetadata()
    
    metadata.setBands(B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11)
    
    tree = ET.parse(MTL)
    root = tree.getroot()
    for child in root:
        if child.tag == 'IMAGE_ATTRIBUTES':
            metadata.setDate(child.find('DATE_ACQUIRED').text, child.find('SCENE_CENTER_TIME').text)
            
        elif child.tag == 'PROJECTION_ATTRIBUTES':
            metadata.setBoundary(float(child.find('CORNER_UL_LAT_PRODUCT').text), 
                                 float(child.find('CORNER_UL_LON_PRODUCT').text),
                                 float(child.find('CORNER_UR_LAT_PRODUCT').text),
                                 float(child.find('CORNER_UR_LON_PRODUCT').text),
                                 float(child.find('CORNER_LL_LAT_PRODUCT').text),
                                 float(child.find('CORNER_LL_LON_PRODUCT').text),
                                 float(child.find('CORNER_LR_LAT_PRODUCT').text),
                                 float(child.find('CORNER_LR_LON_PRODUCT').text),
                                 )
            
        elif child.tag == 'LEVEL1_PROCESSING_RECORD':
            FILE_NAME_BAND_1 = child.find('FILE_NAME_BAND_1').text
            FILE_NAME_BAND_2 = child.find('FILE_NAME_BAND_2').text
            FILE_NAME_BAND_3 = child.find('FILE_NAME_BAND_3').text
            FILE_NAME_BAND_4 = child.find('FILE_NAME_BAND_4').text
            FILE_NAME_BAND_5 = child.find('FILE_NAME_BAND_5').text
            FILE_NAME_BAND_6 = child.find('FILE_NAME_BAND_6').text
            FILE_NAME_BAND_7 = child.find('FILE_NAME_BAND_7').text
            FILE_NAME_BAND_8 = child.find('FILE_NAME_BAND_8').text
            FILE_NAME_BAND_9 = child.find('FILE_NAME_BAND_9').text
            FILE_NAME_BAND_10 = child.find('FILE_NAME_BAND_10').text
            FILE_NAME_BAND_11 = child.find('FILE_NAME_BAND_11').text
            FILE_NAME_BAND_11 = child.find('FILE_NAME_BAND_11').text
            FILE_NAME_METADATA_XML = child.find('FILE_NAME_METADATA_XML').text
            
            
        elif child.tag == 'LEVEL1_THERMAL_CONSTANTS':
            metadata.setKConstants(float(child.find('K1_CONSTANT_BAND_10').text), 
                                 float(child.find('K2_CONSTANT_BAND_10').text),
                                 float(child.find('K1_CONSTANT_BAND_11').text),
                                 float(child.find('K2_CONSTANT_BAND_11').text))

    return metadata







def main(argv):
    landsat_tars_dir = "data/lst"
    stations_file = "data/stations.csv"
    output_dir = 'output'
    arg_input = ""
    arg_output = ""
    arg_user = ""
    arg_help = "{0} -lt <land_tars: Directory containing landsat tars> -st <stations: stations.csv file path> -o <output directory>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hlt:st:o", ["help", "land_tars=", 
        "stations=", "output="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-lt", "--land_tars"):
            landsat_tars_dir = arg
        elif opt in ("-st", "--stations"):
            stations_file = arg
        elif opt in ("-o", "--output"):
            output_dir = arg

    print('landsat_tars_dir:', landsat_tars_dir)
    print('stations_file:', stations_file)
    print('output_dir:', output_dir)
    
    
    tarFiles = []

    stationDataFrame = pd.read_csv(stations_file)

    allStations = []

    for index, row in stationDataFrame.iterrows():
        allStations.append(Station(row['Station ID'],row['Latitude'],row['Longitude']))
            
                    
    getTarFiles(landsat_tars_dir, tarFiles)
    for tarFile in tarFiles[:]:
        name = os.path.basename(tarFile).split('.')[0]
        extractTar(tarFile, TMP_DIR)
        metadata = getTifForImage(TMP_DIR,name)
        metadata.print()
        metadata.calcLST()
        datetimeStr = metadata.DATE_ACQUIRED + " " + metadata.SCENE_CENTER_TIME
        
        inBoundsSize = 0
        for station in allStations:
            lat = station.latitude
            lon = station.longitude
            station.dates.append(datetimeStr)
            val = metadata.getPointVal(lat, lon)
            station.values.append(val)
        for f in os.listdir(TMP_DIR):
            os.remove(TMP_DIR + "/" + f)
            
            
    df = pd.DataFrame(columns=['date','Station ID', 'LST_10', 'LST_11'])
    for station in allStations:
        print(station.station_name)
        for idx, date in enumerate(station.dates):
            print(idx, date)
            newRow = {
                "date": date,
                "Station ID": station.station_name,
                "LST_10": station.values[idx][0],
                "LST_11": station.values[idx][1]
            }
            df = df.append(newRow,ignore_index=True)

    df.to_csv(output_dir + '/lst_result.csv',index=False)

if __name__ == "__main__":
    main(sys.argv)