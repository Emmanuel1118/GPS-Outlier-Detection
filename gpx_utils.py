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


def plot_track_lla(data_array):
    # Extract latitude and longitude
    latitudes = data_array[:, 0].astype(float)
    longitudes = data_array[:, 1].astype(float)
    altitude = data_array[:, 2].astype(float)

    # Plot the track
    plt.figure(figsize=(10, 6))
    #plt.plot(longitudes, latitudes, marker='o', linestyle='-', color='b')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Track')
    plt.grid(True)

    # Add color based on elevation
    scatter = plt.scatter(longitudes, latitudes, c=altitude, cmap='viridis', marker='.', linewidths=2)
    plt.colorbar(scatter, label='Altitude (m)')

    plt.show()


def plot_track_ned(data_array): # curretlly basicaly the same as plot_track_lla
    # Extract NED coordinates
    N = data_array[:, 0].astype(float)
    E = data_array[:, 1].astype(float)
    D = data_array[:, 2].astype(float)

    # Plot the track
    fig = plt.figure(figsize=(12, 8))  # Slightly larger figure for better readability
    ax = fig.add_subplot(111, projection='3d')

    # Set axis labels with better formatting
    ax.set_xlabel('East (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('North (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Height (m)', fontsize=12, labelpad=10)

    # Set title
    plt.title('NED Track 3D Reconstruction', fontsize=15, pad=20)

    # Enhance grid visibility
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Customize scatter plot
    scatter = ax.scatter3D(E, N, D, c=D, cmap='plasma', s=30, alpha=0.8)

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Altitude (m)', fontsize=12)

    # Show the plot
    plt.show()

    # Second Plot
    plt.figure(figsize=(10, 6))
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('NED Track 2D Reconstruction')
    plt.grid(True)

    # Add color based on elevation
    scatter = plt.scatter(E, N, c=D, cmap='viridis', marker='.', linewidths=2)
    plt.colorbar(scatter, label='Altitude (m)')

    plt.show()


def lla2ned(data_array, lla0):
    """
    Convert latitude, longitude, and altitude to North, East, Down (NED) coordinates.

    Parameters:
    data_array (numpy array): Array of latitude, longitude, altitude and time data.
    lla0 (list): Origin latitude, longitude, and altitude.

    Returns:
    ned (numpy array): Array of North, East, Down coordinates, and time.

    Usage Example:
    data_array_ned = lla2ned(data_array_lla, data_array[0, :3])
    """
    # extract origin latitude, longitude, and elevation
    lat0 = np.deg2rad(lla0[0])
    lon0 = np.deg2rad(lla0[1])
    alt0 = lla0[2]

    # extract latitude, longitude, and elevation
    lat = np.deg2rad(data_array[:, 0].astype(float))
    lon = np.deg2rad(data_array[:, 1].astype(float))
    alt = data_array[:, 2].astype(float)

    # radius of the Earth
    R = 6371000 # meters

    # NED coordinates
    north = (R * (lat - lat0))
    east = (R * np.cos(lat0) * (lon - lon0))
    down = alt - alt0

    # stack NED coordinates
    ned = np.column_stack((north, east, down, data_array[:, 3]))
    return ned


def ned2lla(data_array_ned, lla0):
    """
    Convert North, East, Down (NED) coordinates to latitude, longitude, and altitude.
    
    Parameters:
    data_array_ned (numpy array): Array of North, East, Down coordinates and time data.
    lla0 (list): Origin latitude, longitude, and altitude.
    
    Returns:
    lla (numpy array): Array of latitude, longitude, altitude, and time.

    Usage Example:
    data_array_lla = ned2lla(data_array_ned, data_array[0, :3])
    """
    # extract origin latitude, longitude, and elevation
    lat0 = np.deg2rad(lla0[0])
    lon0 = np.deg2rad(lla0[1])
    alt0 = lla0[2]

    # extract NED coordinates
    north = data_array_ned[:, 0].astype(float)
    east = data_array_ned[:, 1].astype(float)
    down = data_array_ned[:, 2].astype(float)

    # radius of the Earth
    R = 6371000 # meters

    # LLA coordinates
    lat = np.rad2deg(lat0 + north / R)
    lon = np.rad2deg(lon0 + east / (R * np.cos(lat0)))
    alt = alt0 + down

    # stack LLA coordinates
    lla = np.column_stack((lat, lon, alt, data_array_ned[:, 3]))
    return lla