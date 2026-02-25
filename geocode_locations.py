"""
Geocode course locations and add lat/lon metadata to the dataset.
Run this once to enrich the data with geographic coordinates.
"""

import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Vague locations to replace with 'TBC'
VAGUE_LOCATIONS = ['District', 'Gardens', 'UK']

def geocode_location(location_name, geolocator, retry=3):
    """
    Geocode a location name to lat/lon coordinates.
    
    Args:
        location_name: Name of the location
        geolocator: Nominatim geolocator instance
        retry: Number of retry attempts
    
    Returns:
        tuple: (latitude, longitude) or (None, None) if failed
    """
    if not location_name or location_name == 'TBC':
        return None, None
    
    # Add UK context for better results
    query = f"{location_name}, UK"
    
    for attempt in range(retry):
        try:
            print(f"  Geocoding: {query} (attempt {attempt + 1}/{retry})")
            location = geolocator.geocode(query, timeout=10)
            
            if location:
                print(f"    ✓ Found: {location.latitude}, {location.longitude}")
                return location.latitude, location.longitude
            else:
                print(f"    ✗ Not found")
                return None, None
                
        except GeocoderTimedOut:
            print(f"    ⚠ Timeout, retrying...")
            time.sleep(2)
        except GeocoderServiceError as e:
            print(f"    ✗ Service error: {e}")
            return None, None
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return None, None
    
    print(f"    ✗ Failed after {retry} attempts")
    return None, None


def main():
    print("=" * 60)
    print("GEOCODING COURSE LOCATIONS")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_pickle("data/courses.pkl")
    print(f"   Loaded {len(df)} courses")
    
    # Replace vague locations with 'TBC'
    print("\n2. Replacing vague locations with 'TBC'...")
    for vague in VAGUE_LOCATIONS:
        count = (df['location'] == vague).sum()
        if count > 0:
            print(f"   Replacing '{vague}' ({count} courses)")
            df.loc[df['location'] == vague, 'location'] = 'TBC'
    
    # Get unique locations (excluding TBC)
    unique_locations = sorted([loc for loc in df['location'].unique() if loc and loc != 'TBC'])
    print(f"\n3. Found {len(unique_locations)} unique locations to geocode")
    print(f"   Locations: {', '.join(unique_locations)}")
    
    # Initialize geolocator with custom user agent
    print("\n4. Initializing Nominatim geocoder...")
    geolocator = Nominatim(user_agent="dandori_course_app_v1.0")
    
    # Create location cache
    location_cache = {}
    
    print("\n5. Geocoding locations (respecting 1 req/sec rate limit)...")
    for i, loc in enumerate(unique_locations, 1):
        print(f"\n[{i}/{len(unique_locations)}] {loc}")
        lat, lon = geocode_location(loc, geolocator)
        location_cache[loc] = {'lat': lat, 'lon': lon}
        
        # Respect Nominatim rate limit (1 request per second)
        if i < len(unique_locations):
            print("   Waiting 1 second (rate limit)...")
            time.sleep(1)
    
    # Add lat/lon columns to dataframe
    print("\n6. Adding lat/lon columns to dataframe...")
    df['latitude'] = None
    df['longitude'] = None
    
    for idx, row in df.iterrows():
        loc = row['location']
        if loc in location_cache:
            df.at[idx, 'latitude'] = location_cache[loc]['lat']
            df.at[idx, 'longitude'] = location_cache[loc]['lon']
    
    # Summary
    print("\n7. Geocoding summary:")
    total_courses = len(df)
    geocoded = df['latitude'].notna().sum()
    tbc = (df['location'] == 'TBC').sum()
    failed = total_courses - geocoded - tbc
    
    print(f"   Total courses: {total_courses}")
    print(f"   Successfully geocoded: {geocoded} ({geocoded/total_courses*100:.1f}%)")
    print(f"   TBC (no location): {tbc} ({tbc/total_courses*100:.1f}%)")
    print(f"   Failed to geocode: {failed} ({failed/total_courses*100:.1f}%)")
    
    # Show location cache
    print("\n8. Location coordinates:")
    for loc, coords in sorted(location_cache.items()):
        if coords['lat'] and coords['lon']:
            print(f"   {loc:25s} → {coords['lat']:.4f}, {coords['lon']:.4f}")
        else:
            print(f"   {loc:25s} → NOT FOUND")
    
    # Save updated dataframe
    print("\n9. Saving updated dataframe...")
    df.to_pickle("data/courses.pkl")
    print("   ✓ Saved to data/courses.pkl")
    
    # Also save location cache as CSV for reference
    cache_df = pd.DataFrame([
        {'location': loc, 'latitude': coords['lat'], 'longitude': coords['lon']}
        for loc, coords in location_cache.items()
    ])
    cache_df.to_csv("data/location_coordinates.csv", index=False)
    print("   ✓ Saved location cache to data/location_coordinates.csv")
    
    print("\n" + "=" * 60)
    print("GEOCODING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
