import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)

    # Translational and rotational velocities of camera
    v = v_cam[0:3, :]
    w = v_cam[3:6, :]

    for i in range(n):
        # Compute Jacobian 
        J = ibvs_jacobian(K, pts_obs[:, i].reshape(2, 1), 1)
        J_t = J[:, 0:3]
        J_w = J[:, 3:6]

        # Compute A
        A = J_t @ v

        # Compute b
        b = (pts_obs[:, i]-pts_prev[:, i]).reshape(2, 1) - J_w @ w

        # Compute linear least squares
        z, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

        #Compute z estimates
        zs_est[i] = 1/z

    #------------------

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est