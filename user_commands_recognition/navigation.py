import pandas as pd
import serial
import pynmea2
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import navitpy

# File storing saved places
PLACES_FILE = "user_commands_recognition/places.csv"

# 1️⃣ Store & Retrieve Locations

def load_places():
    """Loads saved places from places.csv."""
    try:
        return pd.read_csv(PLACES_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["places", "coordinatesx", "coordinatesy"])

def save_place(label, address):
    """Geocodes the address and saves the place in places.csv."""
    geolocator = Nominatim(user_agent="capstone_navigation")
    location = geolocator.geocode(address)

    if location:
        places_df = load_places()
        new_entry = pd.DataFrame([[label, location.latitude, location.longitude]], 
                                 columns=["places", "coordinatesx", "coordinatesy"])
        places_df = pd.concat([places_df, new_entry], ignore_index=True)
        places_df.to_csv(PLACES_FILE, index=False)
        print(f"{label} is saved into {PLACES_FILE} at ({location.latitude}, {location.longitude})")
        return True
    else:
        print("Could not find the address. Try again.")
        return False

# 2️⃣ Get Live GPS Location

def get_gps_position():
    """Reads real-time GPS data from the Grove GPS module."""
    port = "/dev/ttyUSB0"  # Adjust this based on your Raspberry Pi
    ser = serial.Serial(port, baudrate=9600, timeout=1)

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith("$GPGGA"):  # Standard GPS format
            msg = pynmea2.parse(line)
            return msg.latitude, msg.longitude

# 3️⃣ Calculate Distance

def calculate_distance(current_lat, current_lon, dest_lat, dest_lon):
    """Computes the distance (in km) between current location and destination."""
    return geodesic((current_lat, current_lon), (dest_lat, dest_lon)).km

# 4️⃣ Provide Navigation Directions

def get_navigation_directions(start_lat, start_lon, end_lat, end_lon):
    """Generates real-time turn-by-turn directions using NAVIT."""
    nav = navitpy.Navit()
    nav.set_position(start_lat, start_lon)
    route = nav.get_route(end_lat, end_lon)

    directions = [f"{step['distance']} meters: {step['instruction']}" for step in route]
    return directions

def navigate_to_place(place_label, dest_lat, dest_lon):
    """Updates navigation dynamically as the user moves."""
    print(f"Navigating to {place_label} at ({dest_lat}, {dest_lon})")

    while True:
        current_lat, current_lon = get_gps_position()
        distance = calculate_distance(current_lat, current_lon, dest_lat, dest_lon)

        print(f"Current Location: ({current_lat}, {current_lon})")
        print(f"Distance to {place_label}: {distance:.2f} km")

        if distance < 0.05:  # User has arrived (within 50 meters)
            print(f"Arrived at {place_label}!")
            break

        directions = get_navigation_directions(current_lat, current_lon, dest_lat, dest_lon)
        for direction in directions[:3]:  # Provide next steps
            print(direction)

        time.sleep(10)  # Update every 10 seconds
