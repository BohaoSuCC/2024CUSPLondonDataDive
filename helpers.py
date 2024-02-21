import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

def get_nearest_point_by_road(roads, aq_points, congestion_points, buffer_distance=50, max_distance_between_points=300):
    """
    This function finds the nearest congestion point to each air quality testing point
    within a buffer around each road in roads.

    Attributes:
    roads (GeoDataFrame): A GeoDataFrame of roads
    aq_points (GeoDataFrame): A GeoDataFrame of air quality testing points
    congestion_points (GeoDataFrame): A GeoDataFrame of congestion points
    buffer_distance (int): The distance to create a buffer around each road
    max_distance_between_points (int): The minimum distance between points to consider them close enough

    Returns:
    results (list): A list of the nearest congestion points to each air quality testing point
        Each element in the list is a GeoDataFrame with the geometry matching the air quality testing point,
        but with the added elements from the nearest congestion point.
    
    
    """


    # Create a buffer around each road in camden_roads
    roads['buffer'] = roads.buffer(distance=buffer_distance)  # Adjust the buffer distance as needed

    # Initialize a list to store the results
    results = []

    # Iterate over each road buffer
    for index, road in tqdm(roads.iterrows()):
        # Get the buffer geometry
        buffer_geom = road['buffer']
        
        # Check if there are any points from congestion_camden inside the buffer
        congestion_points_inside_buffer = congestion_points[congestion_points.within(buffer_geom)]
        
        # Check if there are any points from no2_camden inside the buffer
        aq_points_inside_buffer = aq_points[aq_points.within(buffer_geom)]
        
        # If there are points from both congestion_camden and no2_camden inside the buffer
        if not congestion_points_inside_buffer.empty and not aq_points_inside_buffer.empty:
            # Iterate over each no2 point inside the buffer
            for _, aq_point in aq_points_inside_buffer.iterrows():
                
                if any(df['geometry'].isin([aq_point['geometry']]).any() for df in results if 'geometry' in df.columns):
                    break


                # Find the nearest congestion point inside the buffer
                nearest_congestion_point = ckdnearest(pd.DataFrame(aq_point).T, congestion_points_inside_buffer)

                distance = pd.to_numeric(nearest_congestion_point['dist']).iloc[0]

                nearest_congestion_point['dist'] = distance

                if distance <= max_distance_between_points:
                    # Add the nearest congestion point to the results list
                    results.append(nearest_congestion_point)

    return results

def ckdnearest(gdA, gdB):
    """
    Get nearest point from gdB to each point in gdA

    Arguments:
    gdA: geopandas dataframe
    gdB: geopandas dataframe

    Returns:
        gpd.DataFrame: dataframe with nearest point from gdB to each point in gdA
    """

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf