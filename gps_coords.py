from gps import *
import time


def getPositionData(gpsd):
  run_program = True
  while run_program:
    nx = gpsd.next()
    if nx['class'] == 'TPV':
        latitude = getattr(nx,'lat', "Unknown")
        longitude = getattr(nx,'lon', "Unknown")
        time = getattr(nx,'time', "Unknown")
        climb = getattr(nx,'climb', "Unknown")
        speed = getattr(nx, 'speed', "Unknown")
        altitude = getattr(nx,'altMSL', "Unknown")
        longitudeerror = getattr(nx,'epx', "Unknown")
        latitudeerror = getattr(nx,'epy', "Unknown")
        mode = getattr(nx,'mode', "Unknown")
        
        print("The target has been detected at " + str(time) + ", with the following GPS coordinates:")
        print("Latitude: " + str(latitude))
        print("Longitude: " + str(longitude))
        print("\nObviously due to signal delays and refraction within the atmosphere this data has an error of +/- " + str(latitudeerror) + "m for latitude and +/- " + str(longitudeerror) + "m for longitude")
        print("\nAircraft Status at time of detection:")
        print("Altitude: " + str(altitude) + "m above sea level")
        print("Ground Speed: " + str(speed) + "m/s")
        print("Rate of Climb: " + str(climb) + "m/s\n")
        print("GPS Status: " + str(mode))
        run_program = False

gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)


getPositionData(gpsd)
