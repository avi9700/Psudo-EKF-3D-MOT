import numpy as np

class HybridEKF:
    def __init__(self, bbox3D, info, ID):
        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1  # Number of total hits including the first detection
        self.info = info  # Other information associated

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

    def adapt_measurement_model(self, source):
        """
        Adjusts the measurement model based on the input source: 
        'deep_learning' or 'clustering'.
        """
        if source == 'deep_learning':
            self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # y
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # z
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # theta
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # l
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # w
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]) # h
        elif source == 'clustering':
            self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # y
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # z
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # l
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # w
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]) # h

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

        # If theta is not available, estimate theta based on the direction of velocity
        if not theta_available:
            vx, vy = self.x[7], self.x[8]
            self.x[3] = np.arctan2(vy, vx)

        # Compute the Jacobian of the state transition function at the current state
        F_jacobian = self.jacobian_F(self.x)

        # Predict the error covariance matrix
        self.P = np.dot(np.dot(F_jacobian, self.P), F_jacobian.T) + self.Q

    def update(self, z, source='deep_learning'):
        # Adapt the measurement model based on the source of the detection
        self.adapt_measurement_model(source)

        # Compute the innovation or residual (y)
        if source == 'deep_learning':
            y = z - self.measurement_function(self.x)
        else:
            # For clustering, exclude theta from the update
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
        self.P = np.dot(I - np.dot(K, H_jacobianLet's complete the implementation and explanation for the Hybrid Extended Kalman Filter (EKF) designed to handle the variability in input from deep learning and clustering-based detections.

### Completing the EKF Update Process

```python
        # Update the error covariance matrix
        I = np.eye(self.x.shape[0])
        self.P = np.dot(I - np.dot(K, H_jacobian), self.P)

    def measurement_function(self, x):
        """
        Defines the linear measurement function. 
        It's adapted depending on whether we have 'deep_learning' or 'clustering' input.
        """
        return np.dot(self.H, x)

    def jacobian_F(self, x):
        """
        Computes the Jacobian of the state transition function. 
        In this implementation, the state transition is linear, so it remains constant.
        """
        return np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x = x + vx
                         [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # y = y + vy
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # z = z + vz
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # theta
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # l
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # w
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # h
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # vx
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # vy
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) # vz

    def jacobian_H(self, x):
        """
        Computes the Jacobian of the measurement function. 
        It's adapted depending on whether we have 'deep_learning' or 'clustering' input.
        """
        return self.H
