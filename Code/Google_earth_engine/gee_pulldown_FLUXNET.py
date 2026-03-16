import config
import ee
import numpy as np
import requests
import traceback
import pandas as pd
import os

ee.Authenticate()
ee.Initialize(project=config.gee_cred)

input_fp = "/Users/chris/TempWorkSpace/KGML/Data/Other_FLUXNET_data/fluxnet_emissions_HH.csv"
output_folder = "/Users/chris/TempWorkSpace/KGML/Data/MODIS_061625/"
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)



def write_data(bands, input_fp):
    try:
        url = bands.getDownloadUrl({
            'region': window,
#            'scale': scale_m,  # spatial resolution in meters/pixel (MCD43A4.061 is 500m resolution)
            'format': 'Geo_TIFF',  # this is necessary for proper formatting, using python API
            'crs': 'EPSG:3857',  # Web Mercator; meter units
            'dimensions': f"{pixel_dim}x{pixel_dim}",
        })
        response = requests.get(url)
        with open(input_fp + ".tif", 'wb') as fd:
            fd.write(response.content)
    except Exception:
        print(traceback.format_exc())

        
def get_collection(product, band, cent, start, end):    
    MODIS = (ee.ImageCollection(product)
             .filterBounds(cent)
             .filterDate(start, end)  # this seems to work. two consecutive days might have the same data though
             .select(band)
             .map(lambda img: img.clip(cent)))  # Even though filterBounds() reduces the number of images to those that overlap with your region, the images may still extend beyond your desired area. clip() ensures that only the part of each image within your window is kept
    MODIS = MODIS.map(datedist)  # getting dates of images (not all days have data)
    dates_list = MODIS.aggregate_array('DateDist')
    dates_list = dates_list.getInfo()
    for d in range(len(dates_list)):
        new_date = pd.to_datetime(dates_list[d], unit='ms').strftime('%Y-%m-%d')
        dates_list[d] = new_date
    MODIS = MODIS.toBands()
    return MODIS, dates_list


def datedist(img):  # https://medium.com/@riyad.jr359/time-efficient-timeseries-data-extraction-on-google-earth-engine-python-api-873ef4540bd4
 img = ee.Image(img)
 date = img.get('system:time_start')
 return img.set('DateDist', date)




# read metadata - .csv
sites = {}
with open(input_fp) as infile:
    header = infile.readline().strip().split(",")
    idind = header.index("SITE_ID") 
    latind = header.index("LAT")
    lonind = header.index("LON")
    for line in infile:
        newline = line.strip().split(",")
        id_ = newline[idind]
        if id_ not in sites:
            sites[id_] = [newline[latind], newline[lonind]]
#
print("pulling from", len(sites), "sites:", sites.keys())


            

### params
buffer_radius_m = 2500
scale_m = 500



counter = 0
for site in sites:
    counter += 1
    print("site", counter, site)
    lat,lon = sites[site]
    lat,lon = float(lat), float(lon)
    window = ee.Geometry.Point([lon, lat])
    window = window.buffer(buffer_radius_m)  # meters
    window = window.transform('EPSG:3857', maxError=1)  # read about what this does
    pixel_dim = int((buffer_radius_m * 2) / scale_m)  # number of pixels along width/height
    #print(f"Fixed output pixel dimensions: {pixel_dim} x {pixel_dim}")

    # reformat the timestamp                                             
    start_time = "2006-01-01"
    end_time = "2018-12-31"
    start_time = ee.Date(start_time)
    end_time = ee.Date(end_time)

    # reflectance (daily; 16day retrieval): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4#description
    #print("1")
    MODIS_BANDS = ['sur_refl_b01',
                   'sur_refl_b02',
                   'sur_refl_b03',
                   'sur_refl_b04',
                   'sur_refl_b05',
                   'sur_refl_b06',
                   'sur_refl_b07',
                   'QA',
                   ]
    MODIS_NAMES = ['sur_refl_b01',
                   'sur_refl_b02',
                   'sur_refl_b03',
                   'sur_refl_b04',
                   'sur_refl_b05',
                   'sur_refl_b06',
                   'sur_refl_b07',
                   'QA',
                   ]
    for b in range(len(MODIS_BANDS)):
        filename = output_folder + str(site) + "_" + MODIS_NAMES[b]
        if not os.path.isfile(filename + ".tif"):
            MODIS,dates = get_collection('MODIS/061/MOD09A1', MODIS_BANDS[b], window, start_time, end_time)
            write_data(MODIS, filename)
            np.save(filename, np.array(dates))

