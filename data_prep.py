from pathlib import Path
import json

import cv2
import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import Polygon


def mask_to_geojson(mask_path, image_path, label, geo=True):
    # ensure the image_path is Path not string
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    
    if geo:
        # Load mask
        with rio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
    # mask = cv2.flip(mask, 0) # different origin point, hence the vertical flip

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare JSON structure
    json_data = {
        "version": "3.21.1",
        "flags": {},
        "shapes": [],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": image_path.name,
        "imageData": "",
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }

    # Convert contours to JSON format with preserved georeferenced coordinates
    # print(f"contour length: {len(contours)}")
    for i, contour in enumerate(contours):
        # print(f"processing contour number: {i}")
        points = contour.squeeze()

        if geo:
            # Apply the transformation to the points to get georeferenced coordinates
            x_coords, y_coords = rio.transform.xy(transform, points[:, 1], points[:, 0])
            georef_points = [(x, y) for x, y in zip(x_coords, y_coords)]
            shape_data = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": georef_points,
                "shape_type": "polygon",
                "flags": {}
            }
        else:
            shape_data = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points.tolist(),
                "shape_type": "polygon",
                "flags": {}
            }
        json_data["shapes"].append(shape_data)
        # print("contour added to shape")

    # Write JSON to file
    json_file_path = str(image_path).replace(".tif", ".json")
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def json_to_geodataframe(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    # Extract shapes from JSON data
    shapes = json_data["shapes"]
    
    # Convert shapes to GeoDataFrame
    polygons = []
    labels = []
    for shape in shapes:
        if len(shape["points"])>=4:
            points = shape["points"]
            label = shape["label"]
            polygons.append(Polygon(points))
            labels.append(label)
    
    gdf = gpd.GeoDataFrame({'geometry': polygons, 'label': labels})
    
    return gdf

