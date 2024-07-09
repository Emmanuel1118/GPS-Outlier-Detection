import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def parse_gpx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Namespace dictionary for Garmin GPX format
    ns = {
        'default': 'http://www.topografix.com/GPX/1/1',
        'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
        'gpxx': 'http://www.garmin.com/xmlschemas/GpxExtensions/v3'
    }
    
    # Extract track points
    data = []
    for trkpt in root.findall('.//default:trkpt', ns):
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        ele = float(trkpt.find('default:ele', ns).text)
        time = trkpt.find('default:time', ns).text
        data.append([lat, lon, ele, time])
    
    # Convert to numpy array
    data_array = np.array(data, dtype=object)
    return data_array


def plot_track(data_array):
    # Extract latitude and longitude
    latitudes = data_array[:, 0].astype(float)
    longitudes = data_array[:, 1].astype(float)
    elevations = data_array[:, 2].astype(float)

    # Plot the track
    plt.figure(figsize=(10, 6))
    #plt.plot(longitudes, latitudes, marker='o', linestyle='-', color='b')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Track')
    plt.grid(True)

    # Add color based on elevation
    scatter = plt.scatter(longitudes, latitudes, c=elevations, cmap='viridis', marker='.', linewidths=2)
    plt.colorbar(scatter, label='Elevation (m)')

    plt.show()