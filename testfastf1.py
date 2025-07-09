import os, fastf1

# Create a folder to store downloaded data
os.makedirs('f1_cache', exist_ok = True)

# Tell FastF1 to cache data there (so you don't re-download every run)
fastf1.Cache.enable_cache('f1_cache')

from fastf1 import get_session
# A session is a practice (Free Practice (FP) 1-3), qualifying or race weekend session
session = get_session(2024, 'Monaco', 'Q')

# Example: the 2024 Monaco Qualifying
# I just loaded the entire session instead of fishing for specific columns
session.load()

# Next we explore lap data
# All laps as a pandas DataFrame
laps = session.laps

# Look at the columns available
# print(laps.columns)

# We filter to the quickest racing laps (skips out-laps, in-laps, slow laps)
quick_laps = laps.pick_quicklaps()

# View the fastest lap for a given driver (e.g. 'VER' driver)
fastest = quick_laps.pick_drivers('VER').pick_fastest()
print(fastest)

# Get a DataFrame of all telemetry channels of the lap you want
# In this case we want the fastest lap
tel = fastest.get_car_data()

# Add a 'distance' column to plot against track distance
tel = tel.add_distance()

# print(tel[['Distance', 'Speed', 'Throttle', 'Brake']].head())

## Fetch weather, timing (schedule) and event info (results)

# Weather readings as a DataFrame
weather = session.weather_data
print(weather.head())

# Event schedule & results
schedule = fastf1.get_event_schedule(2024)
results = session.results

## Simple plot with fastf1's helpers

from fastf1 import plotting
import matplotlib.pyplot as plt

plotting.setup_mpl() # Applies F1-style colors & fonts

fig, ax = plt.subplots()
ax.plot(tel['Distance'], tel['Speed'], label = 'Speed')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Speed (km/h)')
# plt.show()

## Putting into a notebook or script, another example

import os, fastf1
from fastf1 import get_session, plotting
import matplotlib.pyplot as plt

os.makedirs('f1_cache', exist_ok = True)
fastf1.Cache.enable_cache('f1_cache')
plotting.setup_mpl()

# Load data
session = get_session(2024, 'Silverstone', 'R')
session.load()

# Prepare laps & telemetry
laps = session.laps.pick_quicklaps()
lap = laps.pick_driver('HAM').pick_fastest()
tel = lap.get_car_data().add_distance()

# Plot
fig, ax = plt.subplots(figsize = (8,4))
ax.plot(tel['Distance'], tel['Speed'])
ax.set(title = "Hamilton's Fastest Race Lap", xlabel = 'Distance (m)', ylabel = 'Speed (km/h)')
plt.show()