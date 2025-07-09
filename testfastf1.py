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

