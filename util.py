#!/usr/bin/env python3

import geopandas as gpd  # Geospatial data
import pandas as pd  # Tabular data
import numpy as np  # Linear model
import math
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm

BASE_YEAR = 1940
FUTURE_YEAR = 2100
SUPPORTED_MODELS = ["linear", "double", "sqrt", "BH", "Sunamura"]


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


def predict(df, azimuth_lookup, model="linear", Historic_SLR=0.002, Proj_SLR=0.01):
    """_summary_

    Args:
        df (pd.DataFrame): dataframe with columns: TransectID, Date, Distance, YearsSinceBase
        azimuth_lookup (dict): dict lookup of TransectID to Azimuth
        model (str, optional): a model from SUPPORTED_MODELS. Defaults to "linear".
        Historic_SLR (float, optional): Historic Sea Level Rise, only used for the SQRT and BH models. Defaults to 0.002.
        Proj_SLR (float, optional): Projected Sea Level Rise, only used for the SQRT and BH models. Defaults to 0.01.

    Raises:
        ValueError: if you provide an unsupported model

    Returns:
        pd.DataFrame: resulting prediction points for the year 2100
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model} not supported. Supported models: {SUPPORTED_MODELS}"
        )
    grouped = df.groupby("TransectID")
    results = []
    for group_name, group_data in tqdm(grouped):
        if group_name not in azimuth_lookup.keys():
            continue
        coefficients = np.polyfit(group_data.YearsSinceBase, group_data.Distance, 1)
        slope = coefficients[0]
        if model == "double":
            slope *= 2
        elif model == "sqrt":
            # Walkden and Dickson sqrt relationship
            SQRT = (Proj_SLR / Historic_SLR) ** 0.5
            slope *= SQRT
        elif model == "BH":
            # Bray and Hooke relationship inputs
            Length_AP = 82.1
            Prop = 0.1
            B_Height = 20
            C_Depth = 9.577

            # Bray and Hooke relationship
            X = Prop * (B_Height + C_Depth)
            Y = Length_AP / X
            Z = (Proj_SLR - Historic_SLR) * Y
            slope += Z
        elif model == "Sunamura":
            # Sunamara relationship inputs
            Length_AP = 82.1
            C_Depth = 9.577

            # Bray and Hooke relationship
            X = C_Depth / (slope + Length_AP)
            Y = (Proj_SLR - Historic_SLR) / X
            slope += Y

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
    df = enrich_df(df)
    azimuth_lookup = get_azimuth_dict("Shapefiles/OhaweBeach_TransectLines.shp")
    for model in tqdm(SUPPORTED_MODELS):
        results = predict(df, azimuth_lookup, model)
        polygon = prediction_results_to_polygon(results)
        output_shapefile = f"Projected_Shoreline_Polygons/OhaweBeach_{model}.shp"
        polygon.to_file(output_shapefile, driver="ESRI Shapefile")
        print(f"Wrote {output_shapefile}")
