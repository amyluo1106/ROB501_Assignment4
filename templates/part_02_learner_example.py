import numpy as np
from ibvs_controller import ibvs_controller
from dcm_from_rpy import dcm_from_rpy

# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
t_init = np.array([[-0.2, 0.3, -5.0]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gain = 0.1

# Sanity check the controller output if desired.
# ...

# Run simulation - use known depths.
pts_des = np.array([[1, 2, 3, 4],
                    [5, 6,  7,  8]])

pts_obs = np.array([[2, 3, 4, 5],
                    [4, 5,  6,  7]])

zs = np.array([2, 3, 4, 5])

print(ibvs_controller(K, pts_des, pts_obs, zs, gain))