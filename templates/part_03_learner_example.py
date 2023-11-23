import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
import matplotlib.pyplot as plt

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
C_init = dcm_from_rpy([np.pi/3, -np.pi/4, -np.pi/2])
t_init = np.array([[-0.2, 0.3, -5.0]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gain = 1.0
# [0.1, 0.5, 1, 1.5]:

# Sanity check the controller output if desired.
# ...

# Run simulation - estimate depths.
i = ibvs_simulation(Twc_init, Twc_last, pts, K, gain, True)

print(i)

gain = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
iterations = [118, 39, 21, 14, 10, 16, 25]
plt.figure(2)
plt.plot(gain[:], iterations[:], 'go')
plt.grid(True)
plt.show(block = False)
plt.savefig('iterations.png')