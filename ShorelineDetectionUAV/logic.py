# from glob import glob
import numpy as np

from pprint import pprint

import rasterio
import json, re, itertools, os

import matplotlib.pyplot as plt

from math import sin, cos, sqrt, atan2, radians

import cv2 as cv
from sklearn import preprocessing
from sklearn.cluster import KMeans

from PIL import Image
from PIL.TiffTags import TAGS

# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 

import datacube
dc = datacube.Datacube(app = 'my_app', config = '/home/datacube/.datacube.conf')
from utils.data_cube_utilities.dc_display_map import display_map
import utils.data_cube_utilities.data_access_api as dc_api  

# %matplotlib notebook

import xarray.ufuncs as xu
import shapely
from shapely.geometry import Point
from ast import literal_eval
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

from ipywidgets import Button, HBox, VBox
import ipywidgets as widgets
from IPython.display import clear_output, display
# from IPython.html import widgets # Widget definitions
from IPython.utils.traitlets import Unicode # Used to declare attributes of our widget
from datetime import datetime,timedelta

printCharts = 1


# Functinos definitions
def findNearestPoints(dronLinePoints,satteliteLinePoints):
    minDist= []
    droneMinPoint = []
    for point in satteliteLinePoints:
#         point = literal_eval(point)
        shapelyPoint = Point(float(point[1]),float(point[0]))
        distances = []
        for dronePoint in dronLinePoints:
            shapelyDronePoint = Point(dronePoint[1],dronePoint[0])
            sign = 1
            if point[1] > shapelyDronePoint.x and point[0] > shapelyDronePoint.y:
                sign = -1
            distances.append(shapelyPoint.distance(shapelyDronePoint) * sign)
        minDist.append(min(distances))
        droneMinPoint.append(distances.index(min(distances)))
    return droneMinPoint, minDist

def getTransform(path):
    load = rasterio.open(path)
    return load.transform

def findRMSE(transform, satellite, dronPoints, name):
    satellitePoints = findLinePoints(transform,findShoreline(satellite, name,15,0))
    dronMinPoints, minDist = findNearestPoints(dronPoints,satellitePoints)
    createDataFrame(satellitePoints,dronPoints,dronMinPoints,name)

def findLinePoints(transform, edges):
    linePoints = []
    i = 0
    while i < len(edges[0]):
        j = 0
        while j < len(edges[0][i]):
            if (edges[0][i][j] == 255.0):
                pos = transform * (j,i)
                linePoints.append([pos[0],pos[1]])
            j+=1
        i += 1
    return linePoints

def createDataFrame(satteliteLinePoints, dronPoints, droneMinPoint, name):
    df = pd.DataFrame(columns=["satellite_point", "drone_closest_point", "distance (m)"])
    count = len(satteliteLinePoints)
    sumVal = 0
    i = 0
    while i < len(satteliteLinePoints):
        satPoint = satteliteLinePoints[i]#literal_eval(
        dist = getDistanceInM(dronPoints[droneMinPoint[i]][1],dronPoints[droneMinPoint[i]][0],
                      float(satPoint[1]),float(satPoint[0]))
        sumVal = sumVal + dist * dist
        df = df.append({
                 "satellite_point": Point(float(satPoint[1]),float(satPoint[0])),
                 "drone_closest_point":  Point(dronPoints[droneMinPoint[i]][1],dronPoints[droneMinPoint[i]][0]),
                 "distance (m)":  dist
                  }, ignore_index=True)
        i += 1
    print("RMSE ",name, " ", sqrt(sumVal / count))

def getDistanceInM(lat1Coord, lon1Coord, lat2Coord,lon2Coord):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1Coord)
    lon1 = radians(lon1Coord)
    lat2 = radians(lat2Coord)
    lon2 = radians(lon2Coord)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

#     print("Result (m):", distance * 1000)
    return distance * 1000

def findShoreline(images, title, blurSize, showBaseImage):
    subAcquisitions = list(images)
    acquisitionsVectors = [band.reshape(images[0].shape[0] * images[0].shape[1], 1) for band in subAcquisitions]
    reshapedDatasets = np.array(acquisitionsVectors).reshape(3, images[0].shape[0] * images[0].shape[1]).transpose()
    models = []
    model = KMeans(2, max_iter=30)
    model.fit(reshapedDatasets)
    models.append(model)

    clusters = []
    clusters.append(models[0].predict(reshapedDatasets))

    clusteredImages = [clusterLabels.reshape(subAcquisitions[0].shape).astype("uint8") for clusterLabels in clusters]
    blurredImages = [cv.GaussianBlur(clusteredImage, (blurSize,blurSize), 0) for clusteredImage in clusteredImages]
    if printCharts == 1:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,4))
        ax1.imshow(clusters[0].reshape(subAcquisitions[0].shape), cmap="tab20_r")
        ax1.set_title("Segmented acquisition\n2 clusters")
        ax2.imshow(blurredImages[0])
        ax2.set_title("2014-02-01\nGaussian Blurred Image")
        # ax2.imshow(blurredImages[1])
        # ax2.set_title("2019-07-25\nGaussian Blurred Image")
        plt.show()
        
    rawEdges = [cv.Canny(blurredImage, 2, 5).astype("float").reshape(clusteredImages[0].shape) for blurredImage in blurredImages]

    edges = []
    for edge in rawEdges:
        edge[edge == 0] = np.nan
        edges.append(edge)
    if printCharts == 1:
        plt.figure(figsize=(5,5))
        if showBaseImage == 1:
            plt.imshow(getDataAsRGB(images[0],images[1],images[2]))
            plt.imshow(edges[0], cmap = 'Set3_r')
        plt.imshow(edges[0], cmap = 'Set1')
        plt.title('CoastLine %s' % (title))
        plt.show()
    
    return edges

def displayDataAsRGB(imagesR, imagesG, imagesB):
    BIAS = 1.5
    GAIN = [2.3,2.4,1.4]

    r1 = (imagesR - imagesR.min()) / (imagesR.max()-imagesR.min()) * GAIN[0] * BIAS
    g1 = (imagesG - imagesG.min()) / (imagesG.max()-imagesG.min()) * GAIN[1] * BIAS
    b1 = (imagesB - imagesB.min()) / (imagesB.max()-imagesB.min()) * GAIN[2] * BIAS

    rgbImage1 = np.zeros((imagesR.shape[0],imagesR.shape[1],3))
    rgbImage1[:,:,0]= r1
    rgbImage1[:,:,1] = g1
    rgbImage1[:,:,2] = b1

    fig, (ax1) = plt.subplots(1,1,figsize=(6,4))
    ax1.imshow(rgbImage1);
    ax1.set_title("RGB");
    plt.show()
    
def getDataAsRGB(imagesR, imagesG, imagesB):
    BIAS = 1.5
    GAIN = [2.3,2.4,1.4]

    r1 = (imagesR - imagesR.min()) / (imagesR.max()-imagesR.min()) * GAIN[0] * BIAS
    g1 = (imagesG - imagesG.min()) / (imagesG.max()-imagesG.min()) * GAIN[1] * BIAS
    b1 = (imagesB - imagesB.min()) / (imagesB.max()-imagesB.min()) * GAIN[2] * BIAS

    rgbImage1 = np.zeros((imagesR.shape[0],imagesR.shape[1],3))
    rgbImage1[:,:,0]= r1
    rgbImage1[:,:,1] = g1
    rgbImage1[:,:,2] = b1

    return rgbImage1

def printDatasetInfo(sentinel_dataset):
    

def findSatelliteFromDatacube(boundingBox, minTime, maxTime):
    api = dc_api.DataAccessApi(config = '/home/datacube/.datacube.conf')
    platform = "SENTINEL_2"
    product = "s2_l2a_armenia" 
    latitude_extents  = (boundingBox.bottom, boundingBox.top)
    longitude_extents = (boundingBox.left, boundingBox.right)

    time_extents = (minTime, maxTime)
    sentinel_dataset = dc.load(latitude = latitude_extents,
                          longitude = longitude_extents,
                          platform = platform,
                          time = time_extents,
                          product = product,
                          measurements = ['green','red','blue', 'nir', 'swir1', 'swir2' ])
    printDatasetInfo(sentinel_dataset)
    
    return sentinel_dataset

def plotWI(data, name, axs):
    title = "%s" % (name)
    axs.imshow(preprocessing.StandardScaler().fit_transform(data), cmap="Greys_r")
    axs.set_title(title); axs.set_xticklabels([]); axs.set_yticklabels([])

def createRGBArr(data):
    return np.stack([data, data, data], axis=0)
def createRGBArrMult(dataR,dataG,dataB):
    return np.stack([dataR, dataG, dataB], axis=0)

def calculateWaterIndexes(sentinel_dataset):
    B3 = sentinel_dataset.green
    B4 = sentinel_dataset.red
    B8 = sentinel_dataset.nir
    B11 = sentinel_dataset.swir1
    B12 = sentinel_dataset.swir2

    BandRatio = B3/B8
    McFeeters = (B3 - B8) / (B3 + B8)
    MSAVI12 = (2*B8 + 1 - xu.sqrt((2*B8)*(2*B8) - 8*(B8 - B4))) / 2

    MNDWI1 = (B3 - B11) / (B3 + B11)
    MNDWI2 = (B3 - B12) / (B3 + B12)

    if printCharts == 1:
        N_CHARTS = 5
        axs = range(N_CHARTS)
        fig, axs = plt.subplots(2, 3, figsize=(8,6))
        axs = list(itertools.chain.from_iterable(axs))

        plotWI(BandRatio.isel(time = 0),"BandRatio", axs[0])
        plotWI(McFeeters.isel(time = 0),"McFeeters", axs[1])
        plotWI(MSAVI12.isel(time = 0),"MSAVI12", axs[2])
        plotWI(MNDWI1.isel(time = 0),"MNDWI1", axs[3])
        plotWI(MNDWI2.isel(time = 0),"MNDWI2", axs[4])    
        plt.axis("off"); plt.tight_layout(w_pad=0); plt.show()

        displayDataAsRGB(sentinel_dataset.red.isel(time = 0),sentinel_dataset.green.isel(time = 0),sentinel_dataset.blue.isel(time = 0))
    rgbArr = createRGBArrMult(sentinel_dataset.red.isel(time = 0).data,sentinel_dataset.green.isel(time = 0).data,sentinel_dataset.blue.isel(time = 0).data)
    return rgbArr,createRGBArr(BandRatio.isel(time = 0).data), createRGBArr(McFeeters.isel(time = 0).data), createRGBArr(MSAVI12.isel(time = 0).data),createRGBArr(MNDWI1.isel(time = 0).data),createRGBArr(MNDWI2.isel(time = 0).data)

def initAndLoad(tiffFile, startDate, endDate):
    print("Starting process for TIFF: ", tiffFile, ", in date Range: ", startDate, "-", endDate);
    print("Executing Step 1")
    print("Reading metadata")
    load = rasterio.open(tiffFile)
    images = load.read()
    N_OPTICS_BANDS = 4

    print("Fetching data from Datacube")
    sentinel_data = findSatelliteFromDatacube(load.bounds, startDate, endDate)
    print("Step 1 Done")
    
    print("Executing Step 2")
    imageRGB,BandRatioRGB,McFeetersRGB,MNDWI1RGB, MNDWI2RGB,MSAVI12RGB  = calculateWaterIndexes(sentinel_data)

    print("Step 3 for UAV")    
    dronPoints = findLinePoints(load.transform,findShoreline(images[:-1], "Dron",251,0))
    print("Step 3 for UAV Done")    
    
    transform = getTransform("data/Draxtik_reproj_2.tif")
    
    print("Step 4 for UAV vs RGB Image")
    findRMSE(transform, imageRGB, dronPoints,"Image")
    
    print("Step 4 for UAV vs MSAVI12")
    findRMSE(transform, MSAVI12RGB, dronPoints,"MSAVI12")
    print("Step 4 for UAV vs BandRatio")
    findRMSE(transform, BandRatioRGB, dronPoints,"BandRatio")
    print("Step 4 for UAV vs McFeeters")
    findRMSE(transform, McFeetersRGB, dronPoints,"McFeeters")
    print("Step 4 for UAV vs MNDWI2")
    findRMSE(transform, MNDWI2RGB, dronPoints,"MNDWI2")
    

def show():
    path = widgets.Text(value='data/2018_09_15_Draxtik_WGS84_cliped.tif', disabled=False)

    inputDatePicker = widgets.DatePicker(
        description='Select Input file date',
        disabled=False,
        value = datetime.strptime('2018-09-14', '%Y-%m-%d')
    )

    deltaSlider = widgets.IntSlider(
        value=1,
        min=0,
        max=100,
        step=1,
        description='Delta Days',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    def select_files(b):
        clear_output()
        startTime = inputDatePicker.value - timedelta(days=deltaSlider.value)
        endTime = inputDatePicker.value + timedelta(days=deltaSlider.value)
        initAndLoad(path.value,startTime, endTime)
    
    fileselect = Button(description="Run")
    fileselect.on_click(select_files)
    # display(startDatePicker)

    display(VBox([path, HBox([inputDatePicker,deltaSlider,fileselect])]))
    
def showV2(startTime, endTime, minLat, maxLat, minLon, maxLon):
    initAndLoad(startTime, endTime)