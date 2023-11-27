#!/usr/bin/env python3

import geopandas as gpd  # Geospatial data
import pandas as pd  # Tabular data
import numpy as np  # Linear model
from sklearn.linear_model import LinearRegression
import math
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import rapidfuzz  # Fuzzy string matching
import os
import warnings

warnings.filterwarnings(
    "ignore", message="CRS not set for some of the concatenation inputs"
)
warnings.filterwarnings("ignore", message="invalid value encountered in distance")

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
    point = Point(new_x, new_y)
    assert not point.is_empty
    return point


def fit(df: pd.DataFrame, transect_metadata: dict):
    """_summary_

    Fits linear models for each transect, returns slopes and intercepts

    Args:
        df (pd.DataFrame): dataframe with columns: TransectID, Date, Distance, YearsSinceBase
        transect_metadata (dict): dict lookup of TransectID to Azimuth & group

    Returns:
        dict: dict lookup of TransectID to linear model
    """
    grouped = df.groupby("TransectID")
    results = []
    for group_name, group_data in grouped:
        if group_name not in transect_metadata.keys():
            continue
        linear_model = LinearRegression().fit(
            group_data.YearsSinceBase.to_numpy().reshape(-1, 1), group_data.Distance
        )
        results.append({
            "TransectID": group_name,
            "slope": linear_model.coef_[0],
            "intercept": linear_model.intercept_,
            "group": transect_metadata[group_name]["group"],
        })
    return pd.DataFrame(results)

def predict(
    df: pd.DataFrame,
    linear_models: pd.DataFrame,
    transect_metadata: dict,
    Historic_SLR=0.002,
    Proj_SLR=0.01,
    Length_AP=82.1,
    Prop=0.1,
    B_Height=20,
    C_Depth=9.577,
):
    """_summary_

    Args:
        df (pd.DataFrame): dataframe with columns: TransectID, Date, Distance, YearsSinceBase
        linear_models (pd.DataFrame): dataframe with columns: TransectID, slope, intercept
        transect_metadata (dict): dict lookup of TransectID to Azimuth & group
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
    results = []
    for i, row in linear_models.iterrows():
        transect_ID = row.TransectID
        transect_df = df[df.TransectID == transect_ID]
        latest_row = transect_df[transect_df.Date == transect_df["Date"].max()].iloc[0]
        future_year = int(row.get("FUTURE_YEAR", FUTURE_YEAR))
        result = {
            "TransectID": transect_ID,
            "BaselineID": latest_row.BaselineID,
            "group": row.group,
            "Year": future_year,
            "ocean_point": calculate_new_coordinates(
                latest_row.geometry.x,
                latest_row.geometry.y,
                transect_metadata[transect_ID]["Azimuth"] + 180,
                500,
            ),
        }

        for model in SUPPORTED_MODELS:
            slope = row.slope
            intercept = row.intercept
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
                slope -= Z
            elif model == "Sunamura":
                X = C_Depth / (slope + Length_AP)
                Y = (Proj_SLR - Historic_SLR) / X
                slope -= Y
            else:
                raise ValueError(f"Unsupported model: {model}")

            years_until_future = future_year - BASE_YEAR - latest_row.YearsSinceBase
            predicted_distance = slope * years_until_future + intercept
            result[f"{model}_model_point"] = calculate_new_coordinates(
                latest_row.geometry.x,
                latest_row.geometry.y,
                transect_metadata[transect_ID]["Azimuth"],
                predicted_distance,
            )
            result[f"{model}_model_distance"] = predicted_distance
        results.append(result)
    results = gpd.GeoDataFrame(results)
    return results


def prediction_results_to_polygon(results: gpd.GeoDataFrame):
    polygons = []
    for group_name, group_data in results.groupby(["BaselineID", "group"]):
        if len(group_data) > 1:
            polygons.append(
                Polygon(
                    [*list(group_data.geometry), *list(group_data.ocean_point)[::-1]]
                )
            )
    polygons = gpd.GeoSeries(polygons, crs=2193)
    return polygons


def get_transect_metadata(transect_lines_shapefile: str):
    lines = gpd.read_file(transect_lines_shapefile).set_index("TransectID").sort_index()
    lines["dist_to_neighbour"] = lines.distance(lines.shift(-1))
    breakpoints = lines.dist_to_neighbour[lines.dist_to_neighbour > 11]
    lines["group"] = pd.Series(range(len(breakpoints)), index=breakpoints.index)
    lines["group"] = lines.group.bfill().fillna(len(breakpoints)).astype(int)
    metadata = lines[["Azimuth", "group"]].to_dict(orient="index")
    return metadata


def process_file(index_and_row, moving_average=True):
    i, row = index_and_row

    SLR_rate_column_names = row.index[row.index.str.startswith("Rate SSP")]

    site = row.match
    gdf = gpd.read_file(f"Shapefiles/{site}_intersects.shp")
    gdf.crs = 2193
    gdf = enrich_df(gdf)
    transect_metadata = get_transect_metadata(f"Shapefiles/{site}_TransectLines.shp")
    linear_models = fit(gdf, transect_metadata)
    # Erosion only
    linear_models.loc[linear_models.slope > 0, "slope"] = pd.NA
    if moving_average:
        rolled_slopes = linear_models.groupby("group").slope.rolling(10, min_periods=1).mean().dropna().reset_index(level=0)
        linear_models.slope = rolled_slopes.slope
    linear_models.dropna(inplace=True)
    if site == "ManaBayCliffs":
        print("Flipping")
        for k, v in transect_metadata.items():
            transect_metadata[k]["Azimuth"] = v["Azimuth"] + 180
    for SLR_rate_column_name in SLR_rate_column_names:
        results = predict(gdf, linear_models, transect_metadata, Proj_SLR=row[SLR_rate_column_name])
        for model in SUPPORTED_MODELS:
            results.set_geometry(f"{model}_model_point", inplace=True, crs=2193)
            polygon = prediction_results_to_polygon(results)
            os.makedirs("Projected_Shoreline_Polygons", exist_ok=True)
            if model == "linear":
                output_shapefile = f"Projected_Shoreline_Polygons/{site}_{model}.shp"
            else:
                output_shapefile = f"Projected_Shoreline_Polygons/{site}_{model}_{SLR_rate_column_name}.shp"
            polygon.to_file(output_shapefile, driver="ESRI Shapefile")
        results.to_csv(f"Projected_Shoreline_Polygons/{site}_{SLR_rate_column_name}_results.csv", index=False)


def fuzz_preprocess(filename):
    # When fuzzy matching, ignore these strings
    strings_to_delete = ["_", "Cliffs", "Cliff"]
    for s in strings_to_delete:
        filename = filename.replace(s, "")
    # Case-insensitive
    filename = filename.lower()
    # Ignore extension
    filename = os.path.splitext(filename)[0]
    # Basename only
    filename = os.path.basename(filename)
    return filename


def get_match(filename, choices):
    match, score, index = rapidfuzz.process.extractOne(
        query=filename, choices=choices, processor=fuzz_preprocess
    )
    return match, score


def load_AOIs():
    df = pd.read_excel("AOI_SLR_rates.xlsx")
    file_AOIs = [
        f.replace("_intersects.shp", "")
        for f in os.listdir("Shapefiles")
        if f.endswith("_intersects.shp")
    ]
    df["match"], df["match_score"] = zip(*df.AOI.apply(get_match, choices=file_AOIs))
    return df.sort_values(by="match_score")


def run_all_sequential(moving_average=True):
    df = load_AOIs()
    for f in df.iterrows():
        process_file(f, moving_average=moving_average)


def run_all_parallel():
    df = load_AOIs()
    # Run all models on all AOIs in parallel
    process_map(process_file, df.iterrows(), total=len(df))


if __name__ == "__main__":
    # run_all_sequential(moving_average=True)
    run_all_parallel()
