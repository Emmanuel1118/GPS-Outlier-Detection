import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class KF:

    def __init__(self, H, F, G, Q, R, x_0, P_0):

        """
        Initialize the Kalman Filter.

        Args:
            H (np.ndarray): Measurement matrix.
            F (np.ndarray): State transition matrix.
            G (np.ndarray): Noise transition matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            x_0 (np.ndarray): Initial state mean.
            P_0 (np.ndarray): Initial state covariance.
        """

        self.H = H
        self.F = F
        self.G = G
        self.Q = Q
        self.R = R
        self.x = x_0
        self.P = P_0
        self.discard = False
    
    def predict(self):

        """
        Predict the next state and covariance.
        """
        # Predicted state estimate
        self.x = np.dot(self.F, self.x) 
        # Predicted covariance estimate
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + np.dot(np.dot(self.G, self.Q), self.G.T)
    
    def update(self, z, debug=False):

        """
        Update the state estimate with a new measurement.

        Args:
            z (np.ndarray): Measurement vector.
            debug (boolean): Prints K, y and K dot y 
        """
        # Measurement prediction covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        if S.ndim==0:
            s_inv = 1 / S
        else:
            s_inv = np.linalg.inv(S)

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), s_inv) # Named L in the lecture notes

        # Measurement residual (innovation)
        y = z - np.dot(self.H, self.x)

        # Debugging 
        if debug:
            print("--- H dot x ---")
            print(np.dot(self.H, self.x))
            print("--- z ---")
            print(z)
            print("--- y ---")
            print(y)
            print("--- K ---")
            print(K)
            print("--- K dot y ---")
            print(np.dot(K, y))
            print("--- sigma calculation ---")
            print(np.sqrt((np.dot(np.dot(y.T, s_inv), y).astype(float)) / 3))

        # Likely estimation of mesurments. If it is larger then 3*sigma of the measurment sigma then discard the measurment
        if np.sqrt((np.dot(np.dot(y.T, s_inv), y).astype(float)) / 3) > 3:
            self.x = self.x
            self.P = self.P 
            self.discard = True
            return 
        
        self.discard = False # If the estimation is likely

        # Updated state estimate
        if y.ndim == 1 and y.shape[0] == 1:
            a = K * y
            a = a.reshape((2,1))
            self.x = self.x + a
        else:
            self.x = self.x + np.dot(K, y)
        
        # Updated covariance estimate
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def current_state(self):

        """
        Get the current state estimate.
        
        Returns:
            np.ndarray: Current state estimate.
        """
        #print(self.x.shape)
        return self.x, self.discard
    
    def update_system(self, F, G):

        """
        Update the system matrices F and G. for exemple if they are time depentent and the measurments have varying time diffrences.

        Args:
            F (np.ndarray): Updated state transition matrix.
            G (np.ndarray): Updated noise transition matrix.
        """

        self.F = F
        self.G = G



def kf_3d(data, t_diff, kalman):
    """
    Applies a 3D Kalman filter to the provided data.

    Parameters:
    -----------
    data : np.ndarray
        GPS data array (positions in 3D space).

    t_diff : np.ndarray
        Array of time differences between consecutive data points.

    kalman : object
        An instance of a Kalman filter class with methods `update_system()`, `update()`, and `predict()`.

    Returns:
    --------
    meas_update : np.ndarray
        Updated measurements after applying the Kalman filter.
        
    time_update : np.ndarray
        Predicted states after the Kalman filter time update.

    meas_velocity : np.ndarray
        Updated velocities after applying the Kalman filter.

    time_velocity : np.ndarray
        Predicted velocities after the Kalman filter time update.

    discard_ind : np.ndarray
        Indicator for when data is discarded as a result of GPS interference
    """
    
    # Initialize arrays to store measurement and time updates, velocities and discard indicator
    meas_update = np.zeros_like(data[:, :3])
    time_update = np.zeros_like(data[:, :3])
    meas_velocity = np.zeros_like(data[:, :3])
    time_velocity = np.zeros_like(data[:, :3])
    discard_ind = np.zeros_like(data[:, 0])

    # Iterate through the data points, except the last one
    for i in range(len(data[:, 0]) - 1):

        # Time difference for the current step
        t = t_diff[i + 1]

        # Define the system dynamics matrix F and control matrix G based on time difference
        F = np.array([[1, 0, 0, t, 0, 0],
                      [0, 1, 0, 0, t, 0],
                      [0, 0, 1, 0, 0, t],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        
        G = np.array([[0.5 * t**2, 0, 0], 
                      [0, 0.5 * t**2, 0], 
                      [0, 0, 0.5 * t**2], 
                      [t, 0, 0], 
                      [0, t, 0], 
                      [0, 0, t]])
        
        # Update the Kalman filter with the new system matrices
        kalman.update_system(F, G)
        
        # Perform the measurement update using the current data point
        kalman.update(data[i, :3].reshape(3, 1))
        meas, discard_ind[i] = kalman.current_state()
        meas_update[i, :] = meas[:3, :].reshape(3)
        meas_velocity[i, :] = meas[3:, :].reshape(3)

        # Predict the next state and store the time update
        kalman.predict()
        time, _ = kalman.current_state()
        time_update[i, :] = time[:3, :].reshape(3)
        time_velocity[i, :] = time[3:, :].reshape(3)

    return meas_update, time_update, meas_velocity, time_velocity, discard_ind