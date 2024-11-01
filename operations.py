""" Functions for Attitude Control """
import numpy as np
from pyquaternion import Quaternion
import math

def _ECI_to_body(np_eci, np_b, ns_eci, ns_b_goal):
    """
    Defines the spacecraft attitude using primary and secondary normal vectors. 
    The approach is to exactly align the ECI (Earth-Centered Inertial frame) primary vector to the body-frame primary vector,
    and align the ECI secondary vector to the desired secondary body-frame vector as closely as possible.

    Parameters
    ----------
        np_eci : Numpy array
            Normalized primary vector in ECI frame.
        np_b : Numpy array
            Normalized primary vector in body frame.
        ns_eci : Numpy array
            Normalized secondary vector in ECI frame.
        ns_b_goal : Numpy array
            Normalized desired secondary vector in body frame.
    Returns
    -------
        q_eci_to_b : Quaternion object
            A quaternion that rotates a vector from ECI frame to body frame with secondary target considered.
    """
    # primary rotation axis
    np_axis = np.cross(np_b, np_eci) / np.linalg.norm(np.cross(np_b, np_eci))
    # primary rotation angle
    θp = np.arccos(np.dot(np_b, np_eci))
    
    # quaternion that transforms ECI frame to primary target
    # Note: Quaternion objects return q = [w, x, y, z] = [scalar, vector]
    q_eci_to_p = Quaternion(axis=np_axis, radians=θp)
    
    # Apply primary rotation to the desired secondary target 
    # Active rotation: when the vector is rotated with respect to the coordinate system
    ns_p = (q_eci_to_p.inverse * Quaternion(scalar=0, vector=ns_eci) * q_eci_to_p).vector
    
    # Secondary rotation calculations
    A = np.dot(ns_p, ns_b_goal) - np.dot(np_b, ns_p) * np.dot(np_b, ns_b_goal)
    B = np.dot(np.cross(ns_p, np_b), ns_b_goal)
    d_s = A / np.sqrt(A**2 + B**2)
    
    # Optimal rotation: transform primary to body frame with the consideration of secondary target
    # so the secondary target vector in ECI frame is as close as possible to the secondary target in body frame.
    q_scal = np.sign(B) * np.sqrt((1+d_s)/2)
    q_vec = np_b * np.sqrt((1-d_s)/2)
    q_p_to_b = Quaternion(scalar=q_scal, vector=q_vec)
    
    # quaternion that rotates any ECI vector into the body frame
    q_eci_to_b = q_eci_to_p * q_p_to_b
    return q_eci_to_b

def _separation_angle(n1_eci, n2_b, q_eci_to_b):
    """
    Find the separation angle in body frame between an ECI vector 'n1_eci' and a body vector 'n2_b'.
    
    Parameters
    ----------
        n1_eci : Numpy array
            Normalized ECI vector.
        n2_b : Numpy array
            Normalized body vector.
        q_eci_to_b : Quaternion object
            A quaternion describing the spacecraft attitude.
    Returns
    -------
        angle : float
            In degrees.
    """
    # Converts n1_eci to body frame n1_b
    n1_b = (q_eci_to_b.inverse * Quaternion(scalar=0, vector=n1_eci) * q_eci_to_b).vector
    # Calculates the angle between n1_b and n2_b
    angle = np.rad2deg(np.arccos(np.dot(n1_b, n2_b)))
    return angle


""" Functions for Coordinate Conversions """
import numpy as np

def equi_to_cart(coord):
    """ 
    Convert equitorial coordinates to cartesian vector, both in ECI frame. 

    Parameters
    ----------
        coord : tuple
            Equitorial coordinates in the form (RA, DEC).
    Returns
    -------
        vector : Numpy array
    """
    alpha, delta = np.radians(coord[0]), np.radians(coord[1])
    x = np.cos(alpha) * np.cos(delta)
    y = np.sin(alpha) * np.cos(delta)
    z = np.sin(delta)
    vector = np.array([x, y, z])
    return vector

def cart_to_equi(vector):
    """
    Convert cartesian vector to equitorial coordinates, both in ECI frame. 
    
    Parameters
    ----------
        vector : Numpy array
            Cartesian vector in the form [x, y, z].
    Returns
    -------
        coord : tuple
    """
    x, y, z = vector[0], vector[1], vector[2]
    RA = np.rad2deg(math.atan2(y,x)) # already in [-180°, 180°] range
    DEC = np.rad2deg(math.asin(z))
    coord = (RA, DEC)
    return coord

def hms_to_deg(hr, min, sec):
    """
    Convert Right Ascension HMS J2000 format to degrees in [-180°, 180°] range.
    """
    degrees = (hr + min/60 + sec/3600) * 15
    while degrees >= 180:
        degrees -= 360
    while degrees < -180:
        degrees += 360
    return degrees

def dms_to_deg(deg, arcmin, arcsec):
    """
    Convert Declination DMS J2000 format to degrees.
    """
    if deg < 0:
        degrees = -(abs(deg) + arcmin/60 + arcsec/3600)
    else:
        degrees = deg + arcmin/60 + arcsec/3600
    return degrees

