import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

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


def interactive_plot_ned(data_array):
    north = data_array[:, 0].astype(float)
    east = data_array[:, 1].astype(float)
    down = data_array[:, 2].astype(float)

    # Create the 2D track plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=east,
        y=north,
        mode='markers+lines',
        marker=dict(
            size=5,
            color=down,  # Set color to the down values
            colorscale='Viridis',  # Choose a colorscale
            colorbar=dict(title='Height (Down)'),
            showscale=True
        ),
        name='Track'
    ))

    # Set plot titles and labels
    fig.update_layout(
        title='2D Track Plot',
        xaxis_title='East',
        yaxis_title='North',
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal scaling for both axes
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=800,
        height=800
    )

    # Show the plot
    fig.show()


def interactive_plot_lla(lla_coordinates, downsample_factor=1):
    # Downsample data if downsample_factor is greater than 1
    if downsample_factor > 1:
        lla_coordinates = lla_coordinates[::downsample_factor]
    
    # Convert the array of arrays to a pandas DataFrame
    df = pd.DataFrame(lla_coordinates, columns=['latitude', 'longitude', 'altitude', 'time'])
    
    # Create a scatter plot on a map using WebGL
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=dict(
            size=5,  # Adjust marker size for visibility and performance
            color=df['altitude'],
            colorscale='Viridis',
            colorbar=dict(title='Altitude'),
            showscale=True
        )
    ))

    fig.update_layout(
        title='LLA Track Plot',
        mapbox=dict(
            style="carto-positron",
            zoom=3,
            center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()}
        ),
        width=800,
        height=800
    )

    fig.show()


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


def get_time(ned_data):
    """
    Extract and normalize time from the NED data.

    Parameters:
    ned_data : np.array
        A 2D array where each row contains NED coordinates (North, East, Down) and a timestamp.

    Returns:
    time : np.array
        A 1D array containing the normalized time (in seconds), starting from zero.
    """

    time = ned_data[:, 3].astype('datetime64[s]').astype(int)
    time = time - time[0]

    return time


def get_time_diff(data_array):
    """
    Calculate time differences between consecutive timestamps.

    Parameters:
    data_array (numpy array): Array where the 4th column contains datetime values as strings.

    Returns:
    numpy array: Array of time differences in seconds, with the first element as 0.

    Usage Example:
    time_diffs = get_time_diff(data_array)
    """
    # Convert the 4th column of data_array to datetime64 in seconds, then to integer timestamps
    t = get_time(data_array)
    
    # Compute the difference between consecutive timestamps
    diff = np.diff(t)
    
    # Prepend a 0 to the beginning of the differences array to maintain the original length
    diff = np.concatenate([[0], diff])
    
    return diff


def simulate_gps_error(gpx_data, p, error_len=1):
    """
    Simulates GPS errors by replacing sections of the data with a specific error point.

    Parameters:
    gpx_data : np.ndarray
        GPS data array (latitude, longitude, altitude).
    p : float
        Probability of introducing an error at each point.
    error_len : int, optional
        Length of the error (number of consecutive points to replace). Default is 1.

    Returns:
    error_record : np.ndarray
        Modified GPS data with simulated errors.
    """
    error_point = np.array([33.8208, 35.4883, 0]) # Beirut Airport
    error_record = gpx_data.copy()
    skip_counter = 0

    for i in range(len(error_record[:, 0]) - 1):

        if skip_counter > 0:
            # Decrease the skip counter and continue to the next iteration
            skip_counter -= 1
            continue

        if np.random.rand() < p:
            for j in range(error_len):
                if i + j + 1 > len(error_record[:, 0]) - 1:
                    break
                error_record[i + j + 1, :3] = error_point
            skip_counter = error_len - 1
    
    return error_record


def calc_ned_velocity(ned_data):

    """
    Calculate the velocity in the NED (North, East, Down) coordinate frame from position data.

    Parameters:
    ned_data : np.array
        A 2D array where each row contains the NED coordinates (North, East, Down) and a timestamp.

    Returns:
    ned_velocity : np.array
        A 2D array where each row contains the velocity components (North, East, Down) calculated 
        from the position data, with the same number of rows as ned_data.
    """
    ned_velocity = np.zeros([len(ned_data[:, 0]), 3])
    t_diff = get_time_diff(ned_data) # gets the time diffrence between each point

    for i in range(len(ned_data[:, 0]) - 1):

        if t_diff[i+1] == 0:
            if i != 0:
                ned_velocity[i, :] = ned_velocity[i-1, :]
        
        else:
            ned_velocity[i, :] = (ned_data[i+1, :3] - ned_data[i, :3]) / t_diff[i+1]
    
    return ned_velocity


def plot_ned_velocity(velocity, time_stamps):
    """
    Plot the NED (North, East, Down) velocity components over time.

    Parameters:
    velocity : np.array
        A 2D array where each row contains the velocity components (North, East, Down) over time.
    time_stamps : np.array
        A 1D array containing the timestamps corresponding to the velocity data.
    """
    
    # Extract North, East, and Down components of the velocity
    north_velocity = velocity[:, 0]
    east_velocity = velocity[:, 1]
    down_velocity = velocity[:, 2]

    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot the North velocity component
    plt.subplot(3, 1, 1)
    plt.plot(time_stamps, north_velocity, label='North Velocity', color='b')
    plt.ylabel('Velocity (m/s)')
    plt.title('North Velocity')
    plt.grid(True)

    # Plot the East velocity component
    plt.subplot(3, 1, 2)
    plt.plot(time_stamps, east_velocity, label='East Velocity', color='g')
    plt.ylabel('Velocity (m/s)')
    plt.title('East Velocity')
    plt.grid(True)

    # Plot the Down velocity component
    plt.subplot(3, 1, 3)
    plt.plot(time_stamps, down_velocity, label='Down Velocity', color='r')
    plt.ylabel('Velocity (m/s)')
    plt.title('Down Velocity')
    plt.xlabel('Time (s)')
    plt.grid(True)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def avg_discard_distance(data, discard_ind):
    """
    Calculate the average distance between consecutive data points, considering only the discarded points.

    Parameters:
    data : np.array
        A 2D array where each row represents a point in space (e.g., NED coordinates or other vectors).
    discard_ind : np.array or list
        A 1D boolean array or list indicating which points are considered 'discarded' (1 for discarded, 0 for kept).

    Returns:
    avg_distance : float
        The average Euclidean distance between consecutive discarded data points.
    """
       
    norm = sum(discard_ind)
    summ = 0
    for i in range(len(discard_ind)):
        if i:
            summ = summ + np.linalg.norm(data[i, :] - data[i-1, :])
        
    return summ / norm


def is_success_kalman(ned_data):
    return np.max(np.abs(ned_data)) < 100000
