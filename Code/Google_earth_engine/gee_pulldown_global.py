import time
import random
s = random.random() * 300
time.sleep(s)  # Pause to allow multiple runs to start simultaneously (otherwise the ee calls collide)
import ee
import numpy as np
import requests
import traceback
import os
import sys
import rasterio
from datetime import datetime, timedelta
import torch  # for loading data
import gc
import time
import config

ee.Authenticate()
ee.Initialize(config.gee_cred)

rep=int(sys.argv[1])
output_folder = config.wd + "/Out/MODIS_global/"
TEM_preprocess_path = "/Out/prep_TEM.sav"

data0 = torch.load(TEM_preprocess_path, weights_only=False)
Z_sim = data0['Z']
Z_vars_sim = data0['Z_vars']
coords = np.nanmax(Z_sim, axis=1)
del data0
gc.collect()
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# shuffle—consistently across reps using same random seed
shuffled_indices = np.arange(len(coords))
rng = np.random.default_rng(seed=123)
rng.shuffle(shuffled_indices)

# subset table for current rep
max_concurrent_requests = 40
n = int(np.ceil(len(coords) / max_concurrent_requests))   # 61456/40=1537; 12519/40=313

### params
scale_m = 500
start_time = ee.Date("2006-01-01")
end_time = ee.Date("2018-12-31")
product = 'MODIS/061/MOD09A1' # reflectance (daily; 16day retrieval): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4#description
bands =        ['sur_refl_b01',
               'sur_refl_b02',
               'sur_refl_b03',
               'sur_refl_b04',
               'sur_refl_b05',
               'sur_refl_b06',
               'sur_refl_b07',
               ]
lat_ind = list(Z_vars_sim).index("lat")
lon_ind = list(Z_vars_sim).index("long")

### loop through sites and pull down data
image_col = ee.ImageCollection(product)
for i in range(rep*n, (rep*n)+n):
    if i < len(coords):  # not all i's exist
        site = shuffled_indices[i]
        print("site", site, flush=True)
        folder = output_folder + "site_" + str(site)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # check if all bands exist already—if so, avoid calling getInfo() which is expensive
        count=0
        for band in range(len(bands)):
            filename = folder + "/" + bands[band] + ".tif"
            if os.path.exists(filename):
                count+=1
        if count == len(bands):
            continue

        lon = coords[site][lon_ind]
        lat = coords[site][lat_ind]
        bbox = ee.Geometry.Rectangle([lon, lat, lon + 0.5, lat + 0.5])  # xMin, yMin, xMax, yMax]
        coordinates = bbox.getInfo()['coordinates']
        image_clipped = image_col.map(lambda img: img.clip(bbox))  # clipping early hoping to speed things up, but might not help       

        for band in range(len(bands)):
            filename = folder + "/" + bands[band] + ".tif"
            if os.path.exists(filename):
                pass
            else:
                all_months = []
                for year in range(2006, 2018+1):
                    for month in range(1, 12+1):
                        start_date = datetime(year, month, 1)
                        if month == 12:
                            end_date = datetime(year + 1, 1, 1)
                        else:
                            end_date = datetime(year, month + 1, 1)

                        #                                                                                               
                        monthly_col = image_clipped.filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                        def combined_mask(img):
                            qa = img.select('QA')
                            data_band = img.select(bands[band])
                            qa_mask = qa.bitwiseAnd(3).eq(0)  # keeping only pixels with "00" as first two bits           
                            valid_mask = data_band.neq(-28672)  # mask fill values                                        
                            combined = data_band.updateMask(qa_mask.And(valid_mask))
                            return combined
                        monthly_col = monthly_col.map(combined_mask)
                        monthly_col = monthly_col.mean()  # monthly mean AFTER mask—should ignore missing data                                       
                        monthly_col = monthly_col.unmask(-9999)  # missing data value AFTER final mean
                        all_months.append(monthly_col)

                # Combine all monthly images
                multi_band_image = ee.Image(all_months[0])
                for img in all_months[1:]:
                    multi_band_image = multi_band_image.addBands(img)

                # Download single multi-band image
                try:
                    url = multi_band_image.getDownloadURL({
                        'format': 'GeoTIFF',
                        'region': coordinates,
                        'scale': 500,
                        'crs': 'EPSG:4326',
                        'maxPixels': 1e9,
                    })
                    response = requests.get(url)
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                except Exception:
                    print(traceback.format_exc())

                # checking newly downloaded data  
                try:
                    with rasterio.open(filename) as src:
                        data = src.read()
                    print("succesful download", filename, data.shape)
                except Exception as e:
                    print(f"buggy download {filename}: {e}")
                    if os.path.exists(filename):
                        os.remove(filename)

