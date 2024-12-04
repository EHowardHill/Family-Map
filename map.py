from geopy.geocoders import Nominatim
import matplotlib
import json
import os
import numpy as np

matplotlib.use("TkAgg")  # Use 'Qt5Agg' or another if 'TkAgg' doesn't work
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import requests
from fuzzywuzzy import process
from decouple import config

from ged4py import GedcomReader

# Load your Google Maps API key
API_KEY = config("GOOGLE_MAPS_API_KEY")

# Google Maps Geocoding API URL
GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode_location(query):
    """Fetch geocoded data from Google Maps API."""
    params = {"address": query, "key": API_KEY}
    response = requests.get(GEOCODING_API_URL, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            return results
    return []


def resolve_location_name(raw_location):
    """
    Resolve location name to the closest match using Google Maps Geocoding API.
    """
    # Fetch geocoding results for the raw location
    results = geocode_location(raw_location)

    if not results:
        print(f"No geocoding results found for: {raw_location}")
        return raw_location

    # Extract formatted addresses
    formatted_addresses = [result["formatted_address"] for result in results]

    # Find the closest match to the original query
    best_match = process.extractOne(raw_location, formatted_addresses)

    if best_match:
        corrected_location, similarity = best_match
        print(f"Original: {raw_location}")
        print(f"Corrected: {corrected_location} (Similarity: {similarity}%)")
        return corrected_location
    else:
        print(f"No close matches found for: {raw_location}")
        return raw_location


# Initialize geolocator
geolocator = Nominatim(user_agent="family_tree_locator", timeout=10)


# Function to find individual by name
def find_individual_by_name(gedcom_doc, first_name, last_name):
    for individual in gedcom_doc.records0("INDI"):
        names = [individual.name]
        if names:
            name = names[0]  # Assuming the first name is the primary name
            given_name = name.given
            surname = name.surname
            if given_name and surname:
                # Remove slashes from surname if present
                surname = surname.replace("/", "")
                if (
                    given_name.strip().lower() == first_name.lower()
                    and surname.strip().lower() == last_name.lower()
                ):
                    return individual
    return None


# Function to get ancestors up to max_generations
def get_ancestors(individual):
    ancestors = []
    queue = [(individual, 0)]  # (person, generation)
    visited = set()

    while queue:
        person, generation = queue.pop(0)
        if person.xref_id in visited:
            continue
        visited.add(person.xref_id)
        ancestors.append((person, generation))

        # Get parent families
        father = person.father
        mother = person.mother
        if father:
            queue.append((father, generation + 1))
        if mother:
            queue.append((mother, generation + 1))
    return ancestors


# Function to get birth place
def get_birth_place(individual):
    birth = individual.sub_tag("BIRT")
    if birth:
        place = birth.sub_tag_value("PLAC")
        if place:
            return place
    return None


# Functions to map generation to size and redness
def map_generation_to_size(gen, max_gen, min_size, max_size):
    return max_size - (gen / max_gen) * (max_size - min_size)


def map_generation_to_redness(gen, max_gen, min_red, max_red):
    return max_red - (gen / max_gen) * (max_red - min_red)


# Function to serialize birthplaces data
def serialize_birthplaces(birthplaces):
    # Convert all items to serializable data
    serializable_birthplaces = []
    for lat, lon, gen, label in birthplaces:
        serializable_birthplaces.append(
            {"latitude": lat, "longitude": lon, "generation": gen, "label": label}
        )
    return serializable_birthplaces


# Function to deserialize birthplaces data
def deserialize_birthplaces(serializable_birthplaces):
    birthplaces = []
    for item in serializable_birthplaces:
        lat = item["latitude"]
        lon = item["longitude"]
        gen = item["generation"]
        label = item["label"]
        birthplaces.append((lat, lon, gen, label))
    return birthplaces


# Define the data file path
DATA_FILE = "birthplaces.json"
firstname = ""
lastname = ""

with GedcomReader("family.ged") as gedcom_doc:

    # Check if data file exists
    if os.path.exists(DATA_FILE):
        # Load data from file
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            serializable_birthplaces = json.load(f)
        birthplaces = deserialize_birthplaces(serializable_birthplaces)
        print(f"Loaded birthplaces from {DATA_FILE}")
    else:
        # Find yourself
        you = find_individual_by_name(gedcom_doc, firstname, lastname)
        if you is None:
            print("User not found in GEDCOM file.")
            exit()

        # Get ancestors
        ancestors = get_ancestors(you)

        # Initialize geolocator and location cache
        geolocator = Nominatim(user_agent="family_tree_locator", timeout=10)
        location_cache = {}

        # Collect birthplaces and include location names
        birthplaces = []
        for person, generation in ancestors:
            birth_place = get_birth_place(person)
            if birth_place:
                if birth_place in location_cache:
                    location = location_cache[birth_place]
                else:
                    try:
                        location = geolocator.geocode(birth_place)
                        if location is None:
                            try:
                                location = geolocator.geocode(
                                    resolve_location_name(birth_place)
                                )
                            except Exception as e:
                                print(str(e))
                                print(f"Could not geocode place: {birth_place}")
                    except Exception as e:
                        print(f"Geocoding error for place '{birth_place}': {e}")
                        location = None
                    location_cache[birth_place] = location
                if location:
                    latitude = location.latitude
                    longitude = location.longitude
                    # Include the birth_place name in the tuple
                    birthplaces.append((latitude, longitude, generation, birth_place))

        # After processing, save the birthplaces to a file
        serializable_birthplaces = serialize_birthplaces(birthplaces)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable_birthplaces, f, ensure_ascii=False, indent=4)
        print(f"Saved birthplaces to {DATA_FILE}")

# Prepare data for plotting
lats = []
lons = []
sizes = []
colors = []
labels = []
generations = []

# Calculate actual maximum generation
max_gen = max(gen for _, _, gen, _ in birthplaces)

min_size = 20
max_size = 200
min_red = 0.1
max_red = 1.0

for lat, lon, gen, label in birthplaces:
    size = map_generation_to_size(gen, max_gen, min_size, max_size)
    red = map_generation_to_redness(gen, max_gen, min_red, max_red)
    # Ensure red is within the [0, 1] range
    red = max(min(red, max_red), min_red)
    lats.append(lat)
    lons.append(lon)
    sizes.append(size)
    colors.append((red, 0, 0))  # RGB color, varying red channel
    labels.append(label)
    generations.append(gen)

# Convert lists to NumPy arrays for efficient computation
lats = np.array(lats)
lons = np.array(lons)
sizes = np.array(sizes)
colors = np.array(colors)
labels = np.array(labels)

# Plot the world map
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Add state/province lines
ax.add_feature(cfeature.STATES, linestyle=":", alpha=0.5)

# Adjust the map extent as needed
ax.set_global()

# Plot the points with picker enabled
scatter = ax.scatter(
    lons, lats, s=sizes, c=colors, alpha=0.6, transform=ccrs.PlateCarree(), picker=True
)

# Create an empty list to hold text labels
text_labels = []


# Function to update text labels based on current view
def update_labels(event=None):
    # Remove existing text labels
    for txt in text_labels:
        txt.remove()
    text_labels.clear()

    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Determine which points are within the current view
    visible_indices = [
        i
        for i, (lon, lat) in enumerate(zip(lons, lats))
        if xlim[0] <= lon <= xlim[1] and ylim[0] <= lat <= ylim[1]
    ]

    # If fewer than 30 points are visible, add labels
    if len(visible_indices) < 30:
        for i in visible_indices:
            # Slightly offset the label to prevent overlap with the marker
            txt = ax.text(
                lons[i] + 0.2,
                lats[i] + 0.2,
                labels[i],
                fontsize=8,
                transform=ccrs.PlateCarree(),
                ha="left",
                va="bottom",
            )
            text_labels.append(txt)

    # Redraw the figure
    fig.canvas.draw_idle()


# Initially update labels
update_labels()

# Connect the update_labels function to axis limit change events
ax.callbacks.connect("xlim_changed", update_labels)
ax.callbacks.connect("ylim_changed", update_labels)

# Create an annotation for hover display
annot = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(20, 20),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot.set_visible(False)


# Function to update the annotation
def update_annot(ind):
    index = ind["ind"][0]
    pos = lons[index], lats[index]
    annot.xy = pos
    text = labels[index]
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.7)


# Function to handle hover event
def on_hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        # Transform mouse event coordinates to data coordinates
        mouse_lon, mouse_lat = ax.transData.inverted().transform((event.x, event.y))

        # Check if mouse is near any point
        # Calculate distances to all points
        distances = np.sqrt((lons - mouse_lon) ** 2 + (lats - mouse_lat) ** 2)

        # Set a threshold in degrees (adjust as needed)
        threshold = 0.5

        # Find the closest point within the threshold
        if distances.min() <= threshold:
            ind = {"ind": [distances.argmin()]}
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


# Connect hover event
fig.canvas.mpl_connect("motion_notify_event", on_hover)

# Add a color bar to represent generations
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

norm = mcolors.Normalize(vmin=0, vmax=max_gen)
cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "pink"])
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Generations from you")

# Display the map
plt.title("Family Tree Birthplaces")
plt.show()
