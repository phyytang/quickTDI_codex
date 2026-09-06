"""Physical constants and unit conversions for Taiji simulation. """

import math
import numpy as np

# Time units (in seconds) 
YEAR = 24 * 60 * 60 * 365.24  # 365.24 days in seconds 365.24
MONTH = 24 * 60 * 60 * 30
WEEK = 24 * 60 * 60 * 7
DAY = 24 * 60 * 60
HOUR = 60 * 60

# Orbital frequency 
FREQ_ORBIT = 1 / YEAR  # frequency of orbit 

# Speed of light 
SPEED_OF_LIGHT = 2.9979e8  # m/s (speed of light) /
C_NATURAL = 1.0  # speed of light in natural units 

# Arm lengths (in seconds) 
ARM    = 3.000e6 * 1e3 / SPEED_OF_LIGHT  # armlength in seconds - fiducial length  - 
ARM_1  = 3.000e6 * 1e3 / SPEED_OF_LIGHT
ARM_1P = 3.001e6 * 1e3 / SPEED_OF_LIGHT
ARM_2  = 3.010e6 * 1e3 / SPEED_OF_LIGHT
ARM_2P = 3.011e6 * 1e3 / SPEED_OF_LIGHT
ARM_3  = 3.020e6 * 1e3 / SPEED_OF_LIGHT
ARM_3P = 3.021e6 * 1e3 / SPEED_OF_LIGHT

# Mathematical constants 
PI = math.pi

# Astronomical constants 
AU = 149597870700 / SPEED_OF_LIGHT  # Sun-Earth distance in seconds 
R_ORB = AU  # semi-major axis of orbit 


# ============================================================================
# Spacecraft Constellation Structure Constants 
# ============================================================================

# Standard basis vectors 
XVEC = np.array([1, 0, 0])
YVEC = np.array([0, 1, 0])
ZVEC = np.array([0, 0, 1])

# Arm-to-spacecraft mapping dictionaries 
# Maps arm number to receiver spacecraft (arm -> receiver) 
ARM_REC = {1: 2, 2: 3, 3: 1, -1: 3, -2: 1, -3: 2}

# Maps arm number to sender spacecraft (arm -> sender) 
ARM_SEN = {1: 3, 2: 1, 3: 2, -1: 2, -2: 3, -3: 1}

# Maps receiver-sender pair to arm number (RS -> arm) -
RS_ARM = {12: 3, 21: -3, 13: -2, 31: 2, 23: 1, 32: -1}

# Standard arm array ordering for distance arrays 
# Order: [12, 13, 21, 23, 31, 32] -> [3, -2, -3, 1, 2, -1]
# Corresponds to: [1, 1p, 2p, 2, 3, 3p] in matrix form : [1, 1p, 2p, 2, 3, 3p]
ARM_ARRAY = [3, -2, -3, 1, 2, -1]


# ============================================================================
# Backward compatibility: keep the old class-based interface 
# ============================================================================
# This allows existing code to continue working without changes 
class constant:
    """Deprecated: Use module-level constants instead. """
    # Time constants 
    year = YEAR
    month = MONTH
    week = WEEK
    day = DAY
    hour = HOUR

    # Orbital constants 
    forb = FREQ_ORBIT

    # Physical constants 
    SoL = SPEED_OF_LIGHT
    c = C_NATURAL
    pi = PI

    # Arm length constants 
    arm = ARM
    arm1 = ARM_1
    arm1p = ARM_1P
    arm2 = ARM_2
    arm2p = ARM_2P
    arm3 = ARM_3
    arm3p = ARM_3P

    # Astronomical constants 
    au = AU
    Rorb = R_ORB
