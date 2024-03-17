import json

import cv2
import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import Polygon


def mask_to_json(mask_path, image_path, label):
    # Load mask
    with rio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare JSON structure
    json_data = {
        "version": "3.21.1",
        "flags": {},
        "shapes": [],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": image_path,
        "imageData": "",
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }

    # Convert contours to JSON format with preserved georeferenced coordinates
    for contour in contours:
        points = contour.squeeze().tolist()
        # Apply the transformation to the points to get georeferenced coordinates
        georef_points = [rio.transform.xy(transform, p[1], p[0]) for p in points]
        shape_data = {
            "label": label,
            "line_color": None,
            "fill_color": None,
            "points": georef_points,
            "shape_type": "polygon",
            "flags": {}
        }
        json_data["shapes"].append(shape_data)

    # Write JSON to file
    json_file_path = image_path.replace(".tif", ".json")
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    # print("Conversion complete. JSON saved to:", json_file_path)

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

