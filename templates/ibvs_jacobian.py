import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    # Camera intrinsic calibration matrix given by [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
    # The x and y focal lengths in K are guaranteed to be identical.
    f = K[0, 0]
    # Take cx, cy as the principal point
    u_0 = K[0, 2]
    v_0 = K[1, 2]
    # Pixel coordinates of point relative to the principal point
    u_bar = pt[0, 0] - u_0
    v_bar = pt[1, 0] - v_0

    # Compute Jacobian
    J = np.array([[-f/z, 0, u_bar/z, (u_bar*v_bar)/f,  -(f**2+u_bar**2)/f, v_bar],
                  [0, -f/z, v_bar/z, (f**2+v_bar**2)/f, -(u_bar*v_bar/f), -u_bar]])

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J