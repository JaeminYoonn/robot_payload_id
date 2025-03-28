from typing import Tuple

import numpy as np


def to_skew_symmetric(vectors: np.ndarray) -> np.ndarray:
    """
    Converts vectors in R^3 into skew-symmetric form.

    Converts vector(s) v in vectors into skew-symmetric matrix:
        S(v) = -S(v)^T = [[0, -v_3, v_2],
                         [v_3, 0, -v_1],
                         [-v_2, v_1, 0]]

    Args:
        vectors: Vector(s) of shape (..., 3) to convert to matrices.

    Returns:
        Skew-symmetric matrices S(v) of shape (..., 3, 3).
    """
    zero = np.zeros_like(vectors[..., 0])

    row_1 = np.stack((zero, -vectors[..., 2], vectors[..., 1]), axis=-1)
    row_2 = np.stack((vectors[..., 2], zero, -vectors[..., 0]), axis=-1)
    row_3 = np.stack((-vectors[..., 1], vectors[..., 0], zero), axis=-1)

    return np.stack((row_1, row_2, row_3), axis=-2)


def change_inertia_reference_points_with_parallel_axis_theorem(
    I_BBa_B: np.ndarray, m_B: np.ndarray, p_BaBb_B: np.ndarray, Ba_is_Bcm: bool
) -> np.ndarray:
    """
    Converts an inertia matrix represented from one reference point to that
    represented from another reference point.  One of these reference points
    must be the center of mass.

    The parallel axis theorem states [2]:
        I_R = I_C - m_{tot} [d]^2
    for I_C as the inertia matrix about the center of mass, I_R as the moment of
    inertia about a point R defined as R = C + d, and m_{tot} as the total mass of
    the body.  The brackets in [d] indicate the skew-symmetric matrix formed from
    the vector d.

    [2] https://en.wikipedia.org/wiki/Moment_of_inertia#Parallel_axis_theorem

    Args:
        I_BBa_B: Inertia matrices of shape (..., 3, 3).
        m_B: masses of shape (...).
        p_BaBb_B: Displacement from current frame to new frame of shape (..., 3). This
            corresponds to the vector d in the parallel axis theorem.
        Ba_is_Bcm: True if the provided I_BBa_B is from the perspective of the CoM,
            False if from the perspective of the origin.

    Returns:
        Inertia matrices I_BBb_B with changed reference point of shape (..., 3, 3).
    """
    # Check if the input is a single inertia matrix or a batch of inertia matrices
    if I_BBa_B.ndim == 2:
        I_BBa_B = I_BBa_B[np.newaxis, ...]
        m_B = m_B[np.newaxis, ...]
        p_BaBb_B = p_BaBb_B[np.newaxis, ...]
        single_matrix = True
    else:
        single_matrix = False

    d_squared = to_skew_symmetric(p_BaBb_B) @ to_skew_symmetric(p_BaBb_B)
    term = d_squared * m_B.reshape((-1, 1, 1))

    if Ba_is_Bcm:
        I_BBb_B = I_BBa_B - term
    else:
        I_BBb_B = I_BBa_B + term

    if single_matrix:
        return I_BBb_B[0]
    return I_BBb_B


def inertia_to_pseudo_inertia(
    m: float, com: np.ndarray, inertia: np.ndarray
) -> np.ndarray:
    """
    Converts mass, center of mass (com), and inertia matrix to the pseudo-inertia
    matrix.

    Parameters:
    - m (float): Mass of the body.
    - com (np.ndarray): Center of mass vector of shape (3,).
    - inertia (np.ndarray): 3x3 rotational inertia matrix.

    Returns:
    - np.ndarray: The 4x4 pseudo-inertia matrix.
    """
    # Ensure com is a column vector
    com = np.array(com).reshape(3, 1)

    # Density weighted 2nd moment (Σ)
    sigma = 0.5 * np.trace(inertia) * np.eye(3) - inertia

    # Density weighted 1st moment (h)
    h = m * com  # h = m * com (center of mass)

    # Construct the pseudo-inertia matrix
    pseudo_inertia = np.block([[sigma, h], [h.T, np.array([[m]])]])

    return pseudo_inertia


def pseudo_inertia_to_inertia(
    pseudo_inertia: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Converts a pseudo-inertia matrix back to mass, center of mass, and rotational
    inertia matrix.

    Parameters:
    - pseudo_inertia (np.ndarray): 4x4 pseudo-inertia matrix.

    Returns:
    - m (float): Mass of the body.
    - com (np.ndarray): Center of mass vector of shape (3,).
    - inertia (np.ndarray): 3x3 rotational inertia matrix.
    """
    # Extract mass (m)
    m = pseudo_inertia[3, 3]

    # Extract density weighted 1st moment (h)
    h = pseudo_inertia[:3, 3]

    # Center of mass (com)
    com = h / m

    # Extract density weighted 2nd moment (Σ)
    sigma = pseudo_inertia[:3, :3]

    # Rotational inertia (inertia) from Σ
    inertia = np.trace(sigma) * np.eye(3) - sigma

    return m, com, inertia
