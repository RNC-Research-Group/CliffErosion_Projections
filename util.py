#!/usr/bin/env python3

import geopandas as gpd  # Geospatial data
import pandas as pd  # Tabular data
import numpy as np  # Linear model
from sklearn.linear_model import LinearRegression
import math
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm

BASE_YEAR = 1940
FUTURE_YEAR = 2100
SUPPORTED_MODELS = ["linear", "sqrt", "BH", "Sunamura"]


def enrich_df(df: pd.DataFrame):
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


def predict(
    df: pd.DataFrame,
    azimuth_lookup: dict,
    Historic_SLR=0.002,
    Proj_SLR=0.01,
    Length_AP=82.1,
    Prop=0.1,
    B_Height=20,
    C_Depth=9.577
):
    """_summary_

    Args:
        df (pd.DataFrame): dataframe with columns: TransectID, Date, Distance, YearsSinceBase
        azimuth_lookup (dict): dict lookup of TransectID to Azimuth
        Historic_SLR (float, optional): Historic Sea Level Rise, only used for the SQRT, BH and Sunamura models. Defaults to 0.002.
        Proj_SLR (float, optional): Projected Sea Level Rise, only used for the SQRT, BH and Sunamura models. Defaults to 0.01.
        Length_AP (float, optional): Length of active profile, distance between closure depth and cliff toe. Only used for the BH and Sunamura models. Defaults to 82.1.
        Prop (float, optional): Proportion of removable sediments from the beach equilibrium. Only used for the BH model. Defaults to 0.1.
        B_Height (float, optional): Cliff height. Only used for the BH model. Defaults to 20.
        C_Depth (float, optional): Closure depth. Only used for the BH and Sunamura models. Defaults to 9.577.

    Raises:
        ValueError: if you provide an unsupported model

    Returns:
        pd.DataFrame: resulting prediction points for the year 2100
    """
    grouped = df.groupby("TransectID")
    results = []
    for group_name, group_data in grouped:
        if group_name not in azimuth_lookup.keys():
            continue
        linear_model = LinearRegression().fit(group_data.YearsSinceBase.to_numpy().reshape(-1, 1), group_data.Distance)
        slope = linear_model.coef_[0]
        intercept = linear_model.intercept_
        # Erosion only
        if slope > 0:
            continue

        latest_row = group_data[group_data.Date == group_data["Date"].max()].iloc[0]
        result = {
            "TransectID": group_name,
            "Year": FUTURE_YEAR,
            "ocean_point": calculate_new_coordinates(
                latest_row.geometry.x,
                latest_row.geometry.y,
                azimuth_lookup[group_name] + 180,
                500,
            ),
        }

        for model in SUPPORTED_MODELS:
            slope = linear_model.coef_[0]
            intercept = linear_model.intercept_
            if model == "linear":
                pass
            elif model == "sqrt":
                # Walkden and Dickson sqrt relationship
                SQRT = math.sqrt(Proj_SLR / Historic_SLR)
                slope *= SQRT
            elif model == "BH":
                # Bray and Hooke relationship
                X = Prop * (B_Height + C_Depth)
                Y = Length_AP / X
                Z = (Proj_SLR - Historic_SLR) * Y
                slope += Z
            elif model == "Sunamara":
                X = C_Depth / (slope + Length_AP)
                Y = (Proj_SLR - Historic_SLR) / X
                slope += Y

            predicted_distance = slope * (FUTURE_YEAR - BASE_YEAR) + intercept
            distance_difference = latest_row.Distance - predicted_distance
            result[f"{model}_model_point"] = calculate_new_coordinates(
                latest_row.geometry.x,
                latest_row.geometry.y,
                azimuth_lookup[group_name],
                distance_difference,
            )
        results.append(result)
    results = gpd.GeoDataFrame(results)
    return results


def prediction_results_to_polygon(results: gpd.GeoDataFrame):
    polygon = Polygon([*list(results.geometry), *list(results.ocean_point)[::-1]])
    polygon = gpd.GeoSeries(polygon, crs=2193)
    return polygon


def get_azimuth_dict(transect_lines_shapefile: str):
    TransectLine = gpd.read_file(transect_lines_shapefile)
    TransectLine.set_index("TransectID", inplace=True)
    azimuth_lookup = TransectLine.Azimuth.to_dict()
    return azimuth_lookup


if __name__ == "__main__":
    for site in tqdm(["OhaweBeach", "Waitara"]):
        df = gpd.read_file(f"Shapefiles/{site}_intersects.shp")
        df.crs = 2193
        df = enrich_df(df)
        azimuth_lookup = get_azimuth_dict(f"Shapefiles/{site}_TransectLines.shp")
        results = predict(df, azimuth_lookup)
        for model in SUPPORTED_MODELS:
            results.set_geometry(f"{model}_model_point", inplace=True, crs=2193)
            polygon = prediction_results_to_polygon(results)
            output_shapefile = f"Projected_Shoreline_Polygons/{site}_{model}.shp"
            polygon.to_file(output_shapefile, driver="ESRI Shapefile")
            print(f"Wrote {output_shapefile}")
