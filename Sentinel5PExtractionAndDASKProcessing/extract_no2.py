import netCDF4
import os
import pandas as pd

import shapefile

from ARMRegion import ARMRegion
from S5PObject import S5PObject

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def validateItem(item):
    polygon = Polygon(item.region.points)  # create polygon
    for i in range(0, len(item.lat)):
        point = Point(item.lon[i], item.lat[i])  # create point
        if polygon.contains(point):
            item.inside_region.append(True)
        else:
            item.inside_region.append(False)

    return item

def season_of_date(date):
    month = date.month
    if 3 <= month <= 5:
        return 'spring'
    if 6 <= month <= 8:
        return 'summer'
    if 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'
    
    
resultDf = pd.DataFrame(columns=["time","year","month","season","quarter", "point_lat", "point_lon", "region","var_name", "var", "qa_value"])
chemKey = "NO2"

shape = shapefile.Reader("shape/arm_admbnda_adm1_2019.shp")
regions = shape.shapeRecords()

regionObjects = []
for region in regions:
    region = ARMRegion(region.record, region.shape.points, region.shape.bbox)
    regionObjects.append(region)
    
    
    
blacklist = []
files = os.listdir('data')  # storage_data
k = 0
index = 0
no2index = 0
objects = []
for fileName in files:
    if fileName.startswith("S5P_") and ((fileName in blacklist) == False):
        if chemKey in fileName:
            no2index = no2index + 1
            objects.append(S5PObject(fileName, "NO2", "nitrogendioxide_tropospheric_column"))

df = pd.DataFrame(columns=["time", "NO2", "qa_value"])
failedDf = pd.DataFrame(columns=["time", "NO2", "qa_value"])
result_list = []
counter = 0
for x in objects:
    try:
        x.netCDFDataset = netCDF4.Dataset('data/' + x.path, 'r')
        x.setValues()
        x.setValid(True)
        print("Success", x.path)
    except:
        x.setValid(False)
        print("Failed", x.path)
        

items = []
print("days=", len(objects))
for x in objects:
    if x.valid == True:
        for reg in regionObjects:
            item = x.findPointsV2(reg)
            items.append(item)
            
print("points=", len(items))
for i in range(0,len(items)):
    items[i] = validateItem(items[i])
    size = len(items[i].lat)
    print("region=",items[i].region.name,"index=",i, "size=", size)
    for k in range(0,size):
        if items[i].inside_region[k] == True:
            resultDf = resultDf.append({
                "time": items[i].date,
                "year":items[i].date.year,
                "month": items[i].date.month,
                "season": season_of_date(items[i].date),
                "quarter": pd.Timestamp(items[i].date).quarter,
                "point_lat": items[i].lat[k],
                "point_lon": items[i].lon[k],
                "region": items[i].region.name,
                "var_name": chemKey,
                "var": items[i].val[k],
                "qa_value": items[i].qa_value[k]
            },ignore_index=True)
            
print(resultDf)
resultDf.sort_values(by=['time']).to_csv('no2_values.csv', index=False)