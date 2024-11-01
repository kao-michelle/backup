from abc import ABC, abstractmethod

from astropy.coordinates import SkyCoord, GCRS
import astropy.units as u
from astropy.time import Time
from skyfield.api import EarthSatellite, load, wgs84
import numpy as np

from operations import _ECI_to_body, _separation_angle
from windows import Window

class Target(ABC):
    """ 
    Abstract base class for all targets.
    """
    # Class attribute shared by all subclasses
    ephem = load('de421.bsp')
    
    def __init__(self):
        # Note:
        # r_obs_target: the position vector pointing from the observer to the target (in GCRS coordinate system)
        # n_obs_target: the normalized target position vector (i.e. target pointing vector)
        pass

    @abstractmethod
    def pointing(self, satellite=None, time=None):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) frame.
        
        Parameters
        ----------
            satellite : skyfield.sgp4lib.EarthSatellite
            time : Astropy Time object
                The time of observation. Used for determining the position of moving targets.
        Returns
        -------
            n_sat_target : Numpy array
                The normalized vector [x, y, z] in GCRS pointing from the satellite to the target.
        """
        pass

    def _target_bright_angle(self, bright_name, satellite, time):
        """
        The angle (in degrees) between the target and the bright object observed from the satellite.
    
            bright_name : str
                Must be one of 'sun', 'earth', 'moon'. 
            satellite : skyfield.sgp4lib.EarthSatellite
            time : Astropy Time object
        """
        if bright_name not in ['sun', 'earth', 'moon']:
            raise ValueError(f"Invalid 'bright_name': {bright_name}. Must be one of 'sun', 'earth', 'moon'.")
        
        # GCRS position vector from satellite to the bright object
        t = load.timescale().from_astropy(time) # convert to Skyfield time
        r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_geo_bright = self.ephem['earth'].at(t).observe(self.ephem[bright_name]).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_bright = r_geo_bright - r_geo_sat 

        # angle between bright object pointing vector and target pointing vector
        n_sat_bright = r_sat_bright / np.linalg.norm(r_sat_bright)
        n_sat_target = self.pointing(satellite, time)
        angle = np.degrees(np.arccos(np.dot(n_sat_bright, n_sat_target)))
        return angle
    
    def _bright_exclusion_angle(self, bright_name, satellite, time, SEA=10, ELEA=10, MLEA=7):
        """
        The exclusion angles (in degrees) of the bright objects Sun, Earth, and Moon.
            
            bright_name : str
                Must be one of 'sun', 'earth', 'moon'. 
            satellite : skyfield.sgp4lib.EarthSatellite
            time : Astropy Time object
        """
        # convert to Skyfield time
        t = load.timescale().from_astropy(time)
        
        if bright_name == 'sun':
            return SEA
        elif bright_name == 'earth':
            # Earth radius
            Re = 6378.1 # km
            # position vector from Earth to satellite
            r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
            # distance between satellite and Earth
            dist = np.linalg.norm(r_geo_sat) # km
            # angle between Earth's center and limb (Earth Limb Angle) 
            ELA = np.degrees(np.arcsin(Re/dist)) # deg 
            # Earth Exclusion Angle
            EEA = ELA + ELEA # deg 
            return EEA
        elif bright_name == 'moon':
            # Moon radius
            Rm = 1738.1 # km
            # position vector from satellite to moon
            r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
            r_geo_moon = self.ephem['earth'].at(t).observe(self.ephem['moon']).to_skycoord().cartesian.xyz.to(u.km).value # km
            r_sat_moon = r_geo_moon - r_geo_sat # km
            # distance between satellite and Moon
            dist = np.linalg.norm(r_sat_moon) # km
            # angle between Moon's center and limb (Moon Limb Angle) 
            MLA = np.degrees(np.arcsin(Rm/dist)) # deg
            # Moon Exclusion Angle
            MEA = MLA + MLEA # deg
            return MEA
        else:
            raise ValueError(f"Invalid 'bright_name': {bright_name}. Must be one of 'sun', 'earth', 'moon'.")

    def avoid_bright_objects(self, satellite, time, SEA=90, ELEA=10, MLEA=7):
        """
        Checks whether the telescope pointing direction at 'time' avoids Sun, Earth, and Moon. 
    
        Parameters
        ----------
            satellite : skyfield.sgp4lib.EarthSatellite
            time : Astropy Time object
            SEA : float
                Sun exclusion angle with a default value of 90°. 
            ELEA : float
                Earth limb exclusion angle with a default value of 10°. 
            MLEA : float
                Moon limb exclusion angle with a default value of 7°. 
        Returns
        -------
            'True' : bool
                Only when the separation angles between the target and the Sun, Earth, and Moon 
                are all greater than their respective exclusion angles. 
            occulter : list of str
                Otherwise, returns a list of object names that occult the target.
        """
        if isinstance(self, EarthFixedTarget):
            raise TypeError("Does not apply to earth-fixed targets.")
        
        # The separation angle between the target and the bright objects seen from the satellite.
        sun_angle = self._target_bright_angle('sun', satellite, time)
        earth_angle = self._target_bright_angle('earth', satellite, time)
        moon_angle = self._target_bright_angle('moon', satellite, time)
        
        # The bright object exclusion angles.
        SEA = self._bright_exclusion_angle('sun', satellite, time, SEA=SEA)
        EEA = self._bright_exclusion_angle('earth', satellite, time, ELEA=ELEA)
        MEA = self._bright_exclusion_angle('moon', satellite, time, MLEA=MLEA)
    
        # The separation angle must be greater than the exclusion angle. 
        if (sun_angle > SEA) and (earth_angle > EEA) and (moon_angle > MEA): 
            return True
        else:
            occulter = [] # stores the object that occults the target
            if sun_angle <= SEA:
                occulter.append('sun')
            if earth_angle <= EEA:
                occulter.append('earth')
            if moon_angle <= MEA:
                occulter.append('moon')
            return occulter

    def get_visibility_windows(self, start_time, end_time, time_step, satellite, SEA=90, ELEA=10, MLEA=7):
        """ 
        Within the given 'start_time' and 'end_time' period, finds the time periods when the target is visible.
        i.e. Finds the time periods when the target is visible from the 'satellite'.
        
        Parameters
        ----------
            start_time, end_time : Astropy Time objects
                The start and end time of the period for checking target visibility. 
            time_step : int
                The increments (in seconds) to step through the checking period. 
            satellite : skyfield.sgp4lib.EarthSatellite
        Returns
        -------
            windows : Window objects 
        """
        
        if isinstance(self, EarthFixedTarget):
            raise TypeError("Does not apply to earth-fixed targets.")
        
        # Define a time array
        time_range = np.arange(0, (end_time - start_time).to(u.s).value, time_step)
        time_array = start_time + time_range*u.s
    
        visible = []
        occulter = []
        # Iterates through the time array
        for time in time_array:
            # Checks if the satellite is pointing away from bright objects
            avoid_bright_result = self.avoid_bright_objects(satellite, time, SEA, ELEA, MLEA)
    
            # For earth-orbiting targets, also check sunlit_result
            if isinstance(self, EarthOrbitingTarget):
                t = load.timescale().from_astropy(time)
                sunlit_result = satellite.at(t).is_sunlit(self.ephem) # Skyfield positionlib method
                if (avoid_bright_result is True) and (sunlit_result is True):
                    visible.append(True)
                else:
                    visible.append(False)
            # For other target types, just check avoid_bright_result
            else:
                if avoid_bright_result is True:
                    visible.append(True)
                else:
                    visible.append(False)
                    occulter += avoid_bright_result
                    
        # Stores the start and end times of consecutive True sequences
        windows = []
        # Tracks the start time of a consecutive True sequence
        start = None 
        for i, val in enumerate(visible):
            # if the value is True and the start time is not set
            if val==True and start is None:
                # set the start time to be the corresponding time from 'time_array'
                start = time_array[i]
            # if the value is False and the start time is already set
            elif val==False and start is not None:
                end = time_array[i-1] # index back to the most recent True value
                # store the time window
                windows.append((start, end))
                # reset the start time tracker for the next consecutive True sequence
                start = None
        # After iterating through all values, if the start time is still set, that means the list ended with a True
        # then also include this visibility window 
        if start is not None:
            end = time_array[-1] # last element
            windows.append((start, end))
        if len(windows) == 0:
            print(f"The target is occulted by {', '.join(list(set(occulter)))}.")
        return Window(windows)

    # def get_view_efficiency(self, satellite, time):
    #     """
    #     Finds the viewing efficiency of an 'InertialTarget' or a 'SolarSystemTarget'. 
    #     Viewing efficiency is the fraction of the orbit where the boresight is clear of the Earth's limb.

    #     Parameters
    #     ----------
    #         satellite : skyfield.sgp4lib.EarthSatellite
    #         time : Astropy Time object

    #     Returns
    #     -------
    #         view_eff : float
    #             A number from 0 to 1. 
    #     """
    #     if not isinstance(self, InertialTarget) and not isinstance(self, SolarSystemTarget):
    #         raise TypeError("Must be 'InertialTarget' or 'SolarSystemTarget'.")
            
    #     # pointing towards CVZ center
    #     anti_sun_pointing = - SolarSystemTarget('sun').pointing(satellite, time)
    #     target_pointing = self.pointing(satellite, time)
    #     # boresight angle measured from CVZ center
    #     boresight_angle = np.degrees(np.arccos(np.dot(anti_sun_pointing, target_pointing)))
    #     # Earth Exclusion Angle
    #     EEA = self._bright_exclusion_angle('earth', satellite, time)
    #     CVZ_half_angle = 90 - EEA
        
    #     if 0 <= boresight_angle <= CVZ_half_angle:
    #         view_eff = 1
    #     else:
    #         view_eff = 0.5 + (1/np.pi)*np.arcsin((np.cos(np.radians(EEA))/np.sin(np.radians(boresight_angle))))
    #     return view_eff
              
class InertialTarget(Target):
    """
    Distant astronomical targets (e.g. stars, galaxies).
    """
    def __init__(self, ra, dec):
        """
        Parameters
        ----------
            ra, dec : float
                Right ascension (RA) and declination (DEC) equatorial coordinates in degrees.
                RA range is [-180°, +180°] and DEC range is [-90°, +90°].

        Attributes
        ----------
            status : "visible", "other", or "dead"
                "visible" means the target is not occulted.
                "other" means the time when the target is occulted is used for other surveys.
                "dead" means the time when the target is occulted is counted as deadtime. 
            orbit_vis : Window object 
                The target's visibility windows over an orbital period.
            occultation : float
                The time (in seconds) over an orbital period where the target is occulted.
            subexposure_count : int
                Starts at 0; keeps track of the number of subexposures completed
        """
        super().__init__()
        self.ra = ra
        self.dec = dec
        self.skycoord = SkyCoord(ra, dec, frame="icrs", unit="deg").transform_to(GCRS)

        # Initiate 'tile' attributes
        self.status = None
        self.orbit_vis = None
        self.occultation = None
        self.subexposure_count = 0
        self.index = None # assigns the tile index in the order of its initial observation 

    def pointing(self, satellite=None, time=None):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) frame.
        """
        # For distant targets, the pointing vector is approximated as the direction from Earth's center to the target,
        # assuming the target's distance makes satellite's position around Earth negligible.
        n_sat_target = self.skycoord.cartesian.xyz.value
        return n_sat_target

class SolarSystemTarget(Target):
    """
    Solar system objects.
    """
    def __init__(self, name):
        """
        Parameters
        ----------
            name : str
                The name of the ephemeris targets supported by JPL ephemeris DE421.
                See <https://rhodesmill.org/skyfield/planets.html#listing-ephemeris-targets>. 
        """
        super().__init__()
        # Validates ephemeris name
        valid_names = set()
        for names_list in self.ephem.names().values():
            for n in names_list:
                valid_names.add(n.upper()) # convert to uppercase for case-insensitive comparison
        if name.upper() not in valid_names:
            raise ValueError(
                f"'{name}' is not a valid name in the loaded ephemeris." 
                f"Valid names include: {', '.join(sorted(valid_names))}.")
        self.name = name

    def pointing(self, satellite, time):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) frame.
        """
        t = load.timescale().from_astropy(time)
        r_geo_target = self.ephem['earth'].at(t).observe(self.ephem[self.name]).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_target = r_geo_target - r_geo_sat # km
        n_sat_target = r_sat_target / np.linalg.norm(r_sat_target)
        return n_sat_target

class EarthOrbitingTarget(Target):
    """
    Deployed Earth satellites (moving targets).
    """
    def __init__(self, line1, line2):
        """
        Parameters
        ----------
            line1, line2 : str
                The first and second line of the two-line element set (TLE).
        """
        super().__init__()
        self.line1 = line1
        self.line2 = line2
        self.sat = EarthSatellite(line1, line2)

    def pointing(self, satellite, time):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) frame.
        """
        t = load.timescale().from_astropy(time)
        r_geo_target = self.sat.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_target = r_geo_target - r_geo_sat # km
        n_sat_target = r_sat_target / np.linalg.norm(r_sat_target)
        return n_sat_target

class EarthFixedTarget(Target):
    """
    Locations on Earth's surface.
    """
    def __init__(self, lat, lon):
        """
        Parameters
        ----------
        lat, lon : float
            The latitude range is [-90°, +90°] with North as positive.
            The longitude range is [-180°, +180°] with East as positive.
        """
        super().__init__()
        self.lat = lat
        self.lon = lon
    
    def pointing(self, satellite, time):
        """
        Satellite's pointing direction in GCRS, a type of Earth-Centered Inertial (ECI) frame.
        """
        t = load.timescale().from_astropy(time)
        r_geo_target = wgs84.latlon(self.lat, self.lon).at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value # km
        r_sat_target = r_geo_target - r_geo_sat # km
        n_sat_target = r_sat_target / np.linalg.norm(r_sat_target)
        return n_sat_target

    def is_clear_sky(self, time, max_cloud_cover=35, years=10):
        """
        Estimates whether the sky of the target location on a given day in the year is clear enough for optical downlinking. 
        Uses historical cloud coverage data from the Open-Meteo API <https://open-meteo.com/>. 
    
        Parameters
        ----------
            time : Astropy Time object 
                The day interested.
            max_cloud_cover : float 
                Maximum cloud coverage percentage allowed for optical downlinking. Default is 35%.
            years : int
                Number of past years to check historical cloud coverage. Default is 3 years.
        Returns
        -------
            bool
                'True' if the average cloud coverage percentage is below 'max_cloud_cover'. 'False' otherwise.
        """
        from astropy.time import Time
        import requests
        import datetime
        
        # Convert Astropy Time object to a standard datetime.date
        date = time.to_datetime().date()
        month, day = date.month, date.day
        
        # List to store historical cloud coverage for the same month and day
        historical_cloud_covers = []
    
        # Looping over past years to retrieve historical data
        current_year = datetime.date.today().year
        for i in range(1, years + 1):
            year = current_year - i
            try:
                # Handle leap years for February 29th
                if month == 2 and day == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                    continue  # Skip non-leap years
        
                # Build the historical date string
                historical_date_str = f"{year}-{month:02d}-{day:02d}"
               
                # Make the API request to fetch cloud cover for the historical date
                response = requests.get("https://archive-api.open-meteo.com/v1/archive",
                                        params={'latitude': self.lat,
                                                'longitude': self.lon,
                                                'start_date': historical_date_str,
                                                'end_date': historical_date_str,
                                                'hourly': 'cloudcover',
                                                'timezone': 'UTC'}, 
                                        timeout=10) # prevent code from running too long
                response.raise_for_status()
                data = response.json()
                
                # Extract cloud cover data
                cloud_cover = data['hourly']['cloudcover']
                if cloud_cover:
                    # average hourly cloud coverage over the day
                    avg_cloud_cover = sum(cloud_cover) / len(cloud_cover)
                    historical_cloud_covers.append(avg_cloud_cover)
            
            except (requests.exceptions.RequestException, KeyError) as e:
                print(f"Data error for {year}: {e}")
        
        # Calculate average cloud cover over the past years
        if not historical_cloud_covers:
            raise ValueError("Insufficient cloud coverage data. Try increasing 'years'.")
        avg_cloud_cover = sum(historical_cloud_covers) / len(historical_cloud_covers)
        # print(f"Avg cloud coverage: {avg_cloud_cover:.2f}%")
        
        # Determine if the sky is clear
        if avg_cloud_cover <= max_cloud_cover:
            return True
        else:
            return False

    def _find_periods_above_elevation(self, start_time, end_time, elevation_min, satellite):
        """
        Searches between 'start_time' and 'end_time' for the time periods in which the 'satellite' is 
        'elevation_min' degrees above the horizon of the ground station.
        Uses Skyfield sgp4lib find_events(), which returns a tuple (t, events) where the first element 
        is a'skyfield.timelib.Time' in Terrestrial Time and the second element is an array of events:
            * 0 — Satellite rose above 'elevation_min'.
            * 1 — Satellite culminated and started to descend again.
            * 2 — Satellite fell below 'elevation_min'.

        Returns
        -------
            time_periods : list of tuples
                Each time period is ('rising_time', 'falling_time') as Astropy Time objects. 
        """
        # Ground station location
        topos = wgs84.latlon(self.lat, self.lon)
        
        t0 = load.timescale().from_astropy(start_time)
        tf = load.timescale().from_astropy(end_time)
        t, events = satellite.find_events(topos, t0, tf, elevation_min)
        # Skyfield returns 't' in Terrestrial Time (tt)
       
        # Stores the time period when the satellite rises above the given elevation. 
        time_periods = []
        rising_time = None
        for i in range(len(events)):
            if events[i] == 0:  # Satellite rises above elevation
                rising_time = t[i].to_astropy()
                rising_time.format = 'iso'
            elif events[i] == 2 and rising_time is not None:  # Satellite falls below elevation
                falling_time = t[i].to_astropy()
                falling_time.format = 'iso'
                time_periods.append((rising_time.utc, falling_time.utc)) # convert from tt to utc
                rising_time = None
        # Check if the satellite is still above 'elevation_min' at the end of the time array
        if rising_time is not None:
            falling_time = t[-1].to_astropy()
            falling_time.format = 'iso'
            time_periods.append((rising_time.utc, falling_time.utc)) # convert from tt to ut
        return time_periods

    def _downlink_attitude(self, antenna, telescope_boresight, satellite, time):
        """
        Defines the quaternion that points the antenna at Nadir and points the telescope boresight away 
        from the Sun as much as possible.
        """
        t = load.timescale().from_astropy(time)
        r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value
        nadir_pointing = -(r_geo_sat / np.linalg.norm(r_geo_sat))
        anti_sun_pointing = - SolarSystemTarget('sun').pointing(satellite, time)
        q_downlink = _ECI_to_body(nadir_pointing, antenna, anti_sun_pointing, telescope_boresight)
        return q_downlink

    def in_antenna_range(self, pivot_angle, satellite, time):
        """
        Checks whether the target location falls within the antenna range.
        
        Parameters
        ----------
            pivot_angle : float
                The antenna's maximum pivot angle in degrees, i.e. the maximum possible Off-Nadir Angle (ONA). 
        Returns
        -------
            bool
                'True' if the angle between the target and the nadir-pointing antenna is smaller than 'pivot_angle'. 
                'False' otherwise.
        """   
        target_pointing = self.pointing(satellite, time)
        t = load.timescale().from_astropy(time)
        r_geo_sat = satellite.at(t).to_skycoord().cartesian.xyz.to(u.km).value
        nadir_pointing = -(r_geo_sat / np.linalg.norm(r_geo_sat))
        angle = np.degrees(np.arccos(np.dot(target_pointing, nadir_pointing)))
        if angle <= pivot_angle:
            return True
        else:
            return False
        
    def get_downlink_windows(self, start_time, end_time, max_cloud_cover, elevation_min, pivot_angle, antenna, 
                        telescope_boresight, satellite, SEA=90):
        """
        Within the given 'start_time' and 'end_time' period, find the time periods when the satellite can downlink. 
        This requires: (1) a clear sky, (2) the satellite being above 'elevation_min', (3) the ground station being 
        within the antenna 'pivot_angle' range, and (4) the telescope pointing away from the Sun.
    
        Parameters
        ----------
            start_time, end_time : Astropy Time object
                The start and end time of the period for checking target visibility. 
            max_cloud_cover : float 
                Maximum cloud coverage percentage allowed for optical downlinking.
            elevation_min : float
                The minimum ground elevation required to perform downlink.
            pivot_angle : float
                The antenna's maximum pivot angle in degrees, i.e. the maximum possible Off-Nadir Angle (ONA).
            antenna, telescope_boresight : Numpy array
                Spacecraft body-frame unit vectors.
            satellite : skyfield.sgp4lib.EarthSatellite
            SEA : float
                Sun exclusion angle with a default value of 90°. 
        Returns
        -------
            windows : Window objects
        """
        # A list to store downlink window tuples.
        windows = [] 
        
        # Checks whether the sky is clear on the given day.
        if not self.is_clear_sky(start_time, max_cloud_cover):
            return Window()
        
        # Find the time period in which the 'satellite' is 'elevation_min' degrees above the horizon of the ground station.
        above_elevation = self._find_periods_above_elevation(start_time, end_time, elevation_min, satellite)
        
        # Within each 'above_elevation' period 
        for start, end in above_elevation:
            # Define a time array
            time_step = 1 # second
            time_range = np.arange(0, (end - start).to(u.s).value, time_step)
            time_array = start + time_range*u.s
    
            q_t = [] # quaternion profile for angular rate and acceleration calculations
            # To-Do: Check whether the angular rate and acceleration exceeds reaction wheel limitaiton, 
            # and have the ability to amend the slew path if so. 
            
            can_downlink = []
            # With each 'time_step'
            for time in time_array:
                
                # Check if the ground station is in the antenna range. 
                in_range = self.in_antenna_range(pivot_angle, satellite, time)
                
                # Define the downlink attitude
                q_downlink = self._downlink_attitude(antenna, telescope_boresight, satellite, time)
                q_t.append(q_downlink)
                # Find the angle between the telescope and the Sun
                sun_pointing = SolarSystemTarget('sun').pointing(satellite, time)
                angle_from_sun = _separation_angle(sun_pointing, telescope_boresight, q_downlink)
    
                if in_range and angle_from_sun > SEA:
                    can_downlink.append(True)
                else:
                    can_downlink.append(False)
    
            # Tracks the start time of a consecutive True sequence
            start = None 
            for i, val in enumerate(can_downlink):
                # if the value is True and the start time is not set
                if val==True and start is None:
                    # set the start time to be the corresponding time from 'time_array'
                    start = time_array[i]
                # if the value is False and the start time is already set
                elif val==False and start is not None:
                    end = time_array[i-1] # index back to the most recent True value
                    # store the time window
                    windows.append((start, end))
                    # reset the start time tracker for the next consecutive True sequence
                    start = None
            # After iterating through all values, if the start time is still set, that means the list ended with a True
            # then also include this visibility window 
            if start is not None:
                end = time_array[-1] # last element
                windows.append((start, end))
        return Window(windows)