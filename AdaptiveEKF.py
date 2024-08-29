import numpy as np

class AdaptiveEKF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)

        # State vector [x, y, z, theta, l, w, h, vx, vy, vz]
        self.x = np.zeros((10, 1))

        # Initial state covariance matrix (P)
        self.P = np.eye(10) * 10.0
        self.P[7:, 7:] *= 1000.0  # Uncertainty in initial velocity

        # Measurement noise covariance (R)
        self.R = np.eye(7)  # Measurement noise

        # Process noise covariance (Q)
        self.Q = np.eye(10)
        self.Q[7:, 7:] *= 0.01  # Less uncertainty in constant velocity assumption

        # Initialize the state vector with the initial position and size
        self.x[:7] = self.initial_pos.reshape((7, 1))

        # Adaptive process noise factor
        self.adaptive_factor = 1.0

    def adapt_covariances(self):
        """
        Adjust the process and measurement noise covariances based on recent innovation.
        This method should be called periodically.
        """
        # Example heuristic for adapting Q
        innovation_magnitude = np.linalg.norm(self.get_innovation())
        self.adaptive_factor = max(1.0, min(2.0, innovation_magnitude / 10.0))
        self.Q *= self.adaptive_factor
        self.R *= self.adaptive_factor

    def state_transition_function(self, x):
        F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x = x + vx
                      [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # y = y + vy
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # z = z + vz
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # theta = theta
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # l = l (size might change slowly)
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # w = w (size might change slowly)
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # h = h (size might change slowly)
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # vx = vx
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # vy = vy
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) # vz = vz
        return np.dot(F, x)

    def predict(self, theta_available=True):
        # Predict the state using the state transition function
        self.x = self.state_transition_function(self.x)

        # Adapt covariances based on recent performance
        self.adapt_covariances()

        # If theta is not available, estimate theta based on the direction of velocity
        if not theta_available:
            vx, vy = self.x[7], self.x[8]
            self.x[3] = np.arctan2(vy, vx)

        # Compute the Jacobian of the state transition function at the current state
        F_jacobian = self.jacobian_F(self.x)

        # Predict the error covariance matrix
        self.P = np.dot(np.dot(F_jacobian, self.P), F_jacobian.T) + self.Q

    def update(self, z, theta_available=True):
        # Compute the innovation or residual (y)
        if theta_available:
            y = z - self.measurement_function(self.x)
        else:
            y = z[:6] - self.measurement_function(self.x)[:6]

        # Compute the Jacobian of the measurement function at the current state
        H_jacobian = self.jacobian_H(self.x)

        # Compute the innovation covariance (S)
        S = np.dot(H_jacobian, np.dot(self.P, H_jacobian.T)) + self.R

        # Compute the Kalman gain (K)
        K = np.dot(np.dot(self.P, H_jacobian.T), np.linalg.inv(S))

        # Update the state with the new measurement
        self.x = self.x + np.dot(K, y)

        # Update the error covariance matrix
        I = np.eye(self.x.shape[0])
        self.P = np.dot(I - np.dot(K, H_jacobian), self.P)

    def get_innovation(self):
        return np.linalg.norm(self.x[:3] - self.initial_pos[:3])

    def compute_innovation_matrix(self):
        H_jacobian = self.jacobian_H(self.x)
        S = np.dot(H_jacobian, np.dot(self.P, H_jacobian.T)) + self.R
        return S

    def get_velocity(self):
        return self.x[7:]

    def get_size(self):
        return self.x[4:7]



if __name__ == "__main__":
    bbox3D = np.array([0, 0, 0, 0, 1, 1, 1])  # Example initial 3D bounding box
    info = "Object info"  # Example additional info
    ID = 1  # Example ID

    # Create an Adaptive Extended Kalman Filter instance
    ekf = AdaptiveEKF(bbox3D, info, ID)

    # Example measurement update```python
    # Example measurement update with and without theta
    measurement_with_theta = np.array([1, 1, 1, 0.5, 1.2, 1.1, 1.3])
    measurement_without_theta = np.array([1, 1, 1, 1.2, 1.1, 1.3])  # No theta

    # Predict step
    ekf.predict(theta_available=False)  # Predict without theta
    ekf.update(measurement_without_theta, theta_available=False)  # Update without theta

    ekf.predict(theta_available=True)  # Predict with theta
    ekf.update(measurement_with_theta, theta_available=True)  # Update with theta

    # Print updated state
    print("Updated state vector:")
    print(ekf.x)
