#!/usr/bin/env python3

import geopandas as gpd  # Geospatial data
import pandas as pd  # Tabular data
import numpy as np  # Linear model
import math
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm

BASE_YEAR = 1940
FUTURE_YEAR = 2100
SUPPORTED_MODELS = ["linear"]


def enrich_df(df):
    df["Date"] = pd.to_datetime(df.ShorelineI, dayfirst=True)
    df["Year"] = df.Date.dt.year
    df["YearsSinceBase"] = (df.Date - pd.Timestamp(BASE_YEAR, 1, 1)).dt.days / 365.25
    df["YearsUntilFuture"] = (
        pd.Timestamp(FUTURE_YEAR, 1, 1) - df.Date
    ).dt.days / 365.25
    df.Date = df.Date.astype(str)
    return df


def calculate_new_coordinates(old_x, old_y, bearing, distance):
    bearing_radians = math.radians(bearing)
    new_x = old_x + (distance * math.sin(bearing_radians))
    new_y = old_y + (distance * math.cos(bearing_radians))
    return Point(new_x, new_y)


def predict(df, azimuth_lookup, model="linear"):
    df = enrich_df(df)
    grouped = df.groupby("TransectID")
    results = []
    for group_name, group_data in tqdm(grouped):
        if group_name not in azimuth_lookup.keys():
            continue
        coefficients = np.polyfit(group_data.YearsSinceBase, group_data.Distance, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        # Erosion only
        if slope < 0:
            predicted_distance = slope * (FUTURE_YEAR - BASE_YEAR) + intercept
            latest_row = group_data[group_data.Date == group_data["Date"].max()].iloc[0]
            distance_difference = latest_row.Distance - predicted_distance
            results.append(
                {
                    "TransectID": group_name,
                    "Year": FUTURE_YEAR,
                    "Distance": predicted_distance,
                    "geometry": calculate_new_coordinates(
                        latest_row.geometry.x,
                        latest_row.geometry.y,
                        azimuth_lookup[group_name],
                        distance_difference,
                    ),
                    "ocean_point": calculate_new_coordinates(
                        latest_row.geometry.x,
                        latest_row.geometry.y,
                        azimuth_lookup[group_name] + 180,
                        500,
                    ),
                }
            )
    results = gpd.GeoDataFrame(results, crs=2193)
    return results


def prediction_results_to_polygon(results):
    polygon = Polygon([*list(results.geometry), *list(results.ocean_point)[::-1]])
    polygon = gpd.GeoSeries(polygon, crs=2193)
    return polygon


def get_azimuth_dict(transect_lines_shapefile):
    TransectLine = gpd.read_file(transect_lines_shapefile)
    TransectLine.set_index("TransectID", inplace=True)
    azimuth_lookup = TransectLine.Azimuth.to_dict()
    return azimuth_lookup


if __name__ == "__main__":
    df = gpd.read_file("Shapefiles/OhaweBeach_intersects.shp")
    df.crs = 2193
    azimuth_lookup = get_azimuth_dict("Shapefiles/OhaweBeach_TransectLines.shp")
    results = predict(df, azimuth_lookup)
    polygon = prediction_results_to_polygon(results)
    output_shapefile = "Projected_Shoreline_Polygons/OhaweBeach_linear.shp"
    polygon.to_file(output_shapefile, driver="ESRI Shapefile")
    print(f"Wrote {output_shapefile}")
