#!/usr/bin/env python3

from glob import glob
import shutil
import geopandas as gpd  # Geospatial data
import pandas as pd  # Tabular data
import numpy as np  # Linear model
from shapely import LineString
from shapelysmooth import taubin_smooth
from sklearn.linear_model import LinearRegression
import math
from shapely.geometry import Point, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import rapidfuzz  # Fuzzy string matching
from rapidfuzz import process, fuzz
import os
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings(
    "ignore", message="CRS not set for some of the concatenation inputs"
)
warnings.filterwarnings("ignore", message="invalid value encountered in distance")
warnings.filterwarnings("ignore", message="'squared' is deprecated in version 1.4 and will be removed in 1.6")
warnings.filterwarnings("ignore", message="R^2 score is not well-defined with less than two samples")

BASE_YEAR = 1940
FUTURE_YEAR = 2100
SUPPORTED_MODELS = ["linear", "sqrt", "BH", "Sunamura"]


def enrich_df(df: pd.DataFrame):
    df["Date"] = pd.to_datetime(df.ShorelineI, dayfirst=False)
    df["Year"] = df.Date.dt.year
    df["YearsSinceBase"] = (df.Date - pd.Timestamp(BASE_YEAR, 1, 1)).dt.days / 365.25
    df["YearsUntilFuture"] = (
        pd.Timestamp(FUTURE_YEAR, 1, 1) - df.Date
    ).dt.days / 365.25
    df.Date = df.Date.astype(str)
    df["TransectID"] = df.Unique_ID.astype(np.int64)
    return df


def calculate_new_coordinates(old_x, old_y, bearing, distance):
    bearing_radians = math.radians(bearing)
    new_x = old_x + (distance * math.sin(bearing_radians))
    new_y = old_y + (distance * math.cos(bearing_radians))
    point = Point(new_x, new_y)
    assert not point.is_empty
    return point

from rapidfuzz import process, fuzz
choices = ["TransectID"]
process.extractOne ("TransectID", choices)

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
        x = group_data.YearsSinceBase.to_numpy().reshape(-1, 1)
        y = group_data.Distance
        linear_model = LinearRegression().fit(x, y)
        pred = linear_model.predict(x)
        results.append({
            "TransectID": group_name,
            "slope": linear_model.coef_[0],
            "intercept": linear_model.intercept_,
            "group": transect_metadata[group_name]["group"],
            "r2_score": r2_score(y, pred),
            "mae": mean_absolute_error(y, pred),
            "mse": mean_squared_error(y, pred),
            "rmse": mean_squared_error(y, pred, squared=False)
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
        proxy = ",".join(str(p) for p in sorted(transect_df.Proxy.unique()))
        
        latest_row = transect_df[transect_df.Date == transect_df["Date"].max()].iloc[0]
        future_year = int(row.get("FUTURE_YEAR", FUTURE_YEAR))
        result = row.to_dict()
        result.update({
            "TransectID": transect_ID,
            "BaselineID": latest_row.BaselineID,
            "group": row.group,
            "proxy": proxy,
            "Year": future_year,
            "ocean_point": calculate_new_coordinates(
                latest_row.geometry.x,
                latest_row.geometry.y,
                transect_metadata[transect_ID]["Azimuth"] + 180,
                500,
            ),
        })

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

            predicted_distance = slope * (future_year - BASE_YEAR) + intercept
            distance_difference = latest_row.Distance - predicted_distance
            result[f"{model}_model_point"] = calculate_new_coordinates(
                latest_row.geometry.x,
                latest_row.geometry.y,
                transect_metadata[transect_ID]["Azimuth"],
                distance_difference,
            )
            result[f"{model}_model_predicted_distance"] = predicted_distance
            result[f"{model}_model_distance"] = distance_difference
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

#Line and polygon shapefiles are created here 
def prediction_results_to_line_polygon(results: gpd.GeoDataFrame):
    lines = []
    polygons = []
    for group_name, group_data in results.groupby(["BaselineID", "group"]):
        if len(group_data) > 1:
            line = LineString(list(group_data.geometry))
            lines.append(line)
            polygon = Polygon(list(group_data.geometry) + list(group_data.ocean_point)[::-1])
            polygons.append(polygon)
    lines = gpd.GeoSeries(lines, crs=2193)
    polygons = gpd.GeoSeries(polygons, crs=2193)
    return lines, polygons

def prediction_results_to_line_polygon_and_smoothed(results: gpd.GeoDataFrame):
    lines = []
    smoothed_lines = []
    polygons = []
    smoothed_polygons = []
    for group_name, group_data in results.groupby(["BaselineID", "group"]):
        if len(group_data) > 1:
            line = LineString(list(group_data.geometry))
            lines.append(line)
            smoothed_line = taubin_smooth(line, steps=500)
            smoothed_lines.append(smoothed_line)
            polygon = Polygon(list(group_data.geometry) + list(group_data.ocean_point)[::-1])
            polygons.append(polygon)
            smoothed_polygon = Polygon(list(smoothed_line.coords) + list(group_data.ocean_point)[::-1])
            smoothed_polygons.append(smoothed_polygon)
    lines = gpd.GeoSeries(lines, crs=2193)
    smoothed_lines = gpd.GeoSeries(smoothed_lines, crs=2193)
    polygons = gpd.GeoSeries(polygons, crs=2193)
    smoothed_polygons = gpd.GeoSeries(smoothed_polygons, crs=2193)
    return lines, polygons, smoothed_lines, smoothed_polygons


def get_transect_metadata(lines):
    lines["dist_to_neighbour"] = lines.distance(lines.shift(-1))
    breakpoints = lines.dist_to_neighbour[lines.dist_to_neighbour > 20]
    lines["group"] = pd.Series(range(len(breakpoints)), index=breakpoints.index)
    lines["group"] = lines.group.bfill().fillna(len(breakpoints)).astype(int)
    metadata = lines[["Azimuth", "group"]].to_dict(orient="index")
    return metadata

def get_transects(intersects):
  p1 = intersects.geometry[intersects.Distance.idxmin()].coords[0]
  p2 = intersects.geometry[intersects.Distance.idxmax()].coords[0]
  azimuth = math.degrees(math.atan2(p1[0]-p2[0], p1[1]-p2[1]))
  if azimuth < 0:
      azimuth += 360
  return pd.Series({"Azimuth": azimuth, "geometry": LineString([p1, p2])})


def process_file(file, moving_average=True, fix_accretion=True):
    site = os.path.basename(file).replace("_Intersects_Proxy.shp", "")
    gdf = gpd.read_file(file)
    gdf.crs = 2193
    gdf = enrich_df(gdf)
    
    lines = gdf.groupby("TransectID")[["geometry", "Distance"]].apply(get_transects)
    lines.crs = gdf.crs
    transect_metadata = get_transect_metadata(lines)

    linear_models = fit(gdf, transect_metadata)
    linear_models["original_slope"] = linear_models.slope
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
    results = predict(gdf, linear_models, transect_metadata)
    if fix_accretion:
        results = gpd.GeoDataFrame(results[results.linear_model_distance >= 0])
    #Saving line and polygon projection file to folder in VS Code. Change file location accordingly

    for model in SUPPORTED_MODELS:
        results.set_geometry(f"{model}_model_point", inplace=True, crs=2193)
        lines, polygons, smoothed_lines, smoothed_polygons = prediction_results_to_line_polygon_and_smoothed(results)
        os.makedirs("Projections", exist_ok=True)
        polygons.to_file(f"Projections/{site}_{model}_polygon.shp")
        lines.to_file(f"Projections/{site}_{model}_line.shp")

        #Smoothening occurs here. Currently 500 steps of Taubin's algorithm applied. Change the file name and location accordingly.  
        smoothed_lines.to_file(f"Projections/{site}_{model}_line_smoothed.shp")
        smoothed_polygons.to_file(f"Projections/{site}_{model}_polygon_smoothed.shp")

        good_results = results[results.proxy.isin(["1", "2", "3", "4", "5", "6", "0,1", "0,2", "0,3", "0,4", "0,5", "0,6", "2,3", "1,4", "1,5", "1,6", "1,5,6"])]
        
        good_results.set_geometry(f"{model}_model_point", inplace=True, crs=2193)
        lines, polygons, smoothed_lines, smoothed_polygons = prediction_results_to_line_polygon_and_smoothed(good_results)
        if len(good_results) >= 1:
            polygons.to_file(f"Projections_good/{site}_{model}_polygon.shp")
            lines.to_file(f"Projections_good/{site}_{model}_line.shp")
            #Smoothening occurs here. Currently 500 steps of Taubin's algorithm applied. Change the file name and location accordingly.  
            smoothed_lines.to_file(f"Projections_good/{site}_{model}_line_smoothed.shp")
            smoothed_polygons.to_file(f"Projections_good/{site}_{model}_polygon_smoothed.shp")
            

        if model in ["sqrt", "BH", "Sunamura"]:
            good_results = results[results.proxy.isin(["2", "3", "0,2", "0,3", "2,3", "0,2,3"])]
            good_results.set_geometry(f"{model}_model_point", inplace=True, crs=2193)
            lines, polygons, smoothed_lines, smoothed_polygons = prediction_results_to_line_polygon_and_smoothed(good_results)
            if len(good_results) >= 1:
                polygons.to_file(f"Projections_best/{site}_{model}_polygon.shp")
                lines.to_file(f"Projections_best/{site}_{model}_line.shp")
                smoothed_lines.to_file(f"Projections_best/{site}_{model}_line_smoothed.shp")
                smoothed_polygons.to_file(f"Projections_best/{site}_{model}_polygon_smoothed.shp")


    results.to_csv(f"Projections/{site}_results.csv", index=False)


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
    files = sorted(glob(f"Data/Merged Intersects_UniqueID_Proxy/*.shp"))
    for folder in ["Projections", "Projections_good", "Projections_best"]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
    for f in tqdm(files):
        process_file(f)
