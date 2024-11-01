from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec, WGS72
from astropy.time import Time
import numpy as np

class Satellite:
    
    @staticmethod
    def build_from_orbit(T0, ecco, argpo, inclo, RAAN, period, mo):
        """
        Create a Skyfield EarthSatellite object from orbital parameters.
        
        Orbit set-up: The reference plane is the Earth's equitorial plane. The intersection 
        between the reference plane and the orbital plane is the line-of-nodes, connecting 
        the ECI frame origin with the acending and descending nodes.
    
        Parameters
        ----------
            T0 : Astropy Time object
                Epoch time specifying the moment at which the orbital elements are defined. 
            ecco : float
                Eccentricity of the orbit. 
            argpo : float
                Argument of perigee is the angle in degrees measured from the ascending node 
                to the perigee; defines the orientation of the ellipse in the orbital plane.
            inclo : float
                Inclination is the vertical tilt in degrees of the orbital plane with respect 
                to the reference plane, measured at the ascending node.
            RAAN : float
                The right ascension of ascending node is the angle in degrees along the reference plane 
                of the ascending node with respect to the reference frame’s vernal equinox.
                (i.e. horizontally orients the ascending node)
            period : float
                Orbital period in minutes.
            mo : float
                Mean anomaly is the angle in degrees from perigee defining the position of the satellite 
                along the elipse at epoch T0.
        Returns
        -------
            satellite : 'skyfield.sgp4lib.EarthSatellite'
        """
        # Initiate the satellite object
        satrec = Satrec()
    
        # Define the 'epoch' in sgp4 satrec 
        tau0 = Time("1949-12-31 0:0:0").jd 
        tau = T0.jd
        epoch = tau - tau0
        
        # Set up a low-level constructor that builds a satellite model directly from numeric orbital parameters
        # Code reference <https://rhodesmill.org/skyfield/earth-satellites.html>
        # See <https://pypi.org/project/sgp4/#providing-your-own-elements> for more details
        satrec.sgp4init(
            WGS72,                 # gravity model
            'i',                   # 'a' = old AFSPC mode, 'i' = improved mode
            0,                     # satnum: Satellite number (set to 0 for custom satellites)
            epoch,                 # epoch: days since 1949 December 31 00:00 UT
            0.0,                   # bstar: drag coefficient (/earth radii)
            0.0,                   # ndot: ballistic coefficient (revs/day)
            0.0,                   # nddot: second derivative of mean motion (revs/day^3)
            ecco,                  # ecco: eccentricity
            np.radians(argpo),     # argpo: argument of perigee (radians)
            np.radians(inclo),     # inclo: inclination (radians)
            np.radians(mo),        # mo: mean anomaly (radians)
            (2 * np.pi) / period,  # no_kozai: mean motion (radians/minute)
            np.radians(RAAN),      # nodeo: right ascension of ascending node (radians)
        )
        
        # Wrap this low-level satellite model in a Skyfield EarthSatellite object
        ts = load.timescale()
        satellite = EarthSatellite.from_satrec(satrec, ts)
        return satellite

    @staticmethod
    def build_from_TLE(line1, line2):
        """
        Create a Skyfield EarthSatellite object from Two-Line Element (TLE) data.

        Parameters
        ----------
            line1, line2 : str
                The first and second line of the two-line element set (TLE).
        Returns
        -------
            satellite : 'skyfield.sgp4lib.EarthSatellite'
        """
        satellite = EarthSatellite(line1, line2)
        return satellite

        