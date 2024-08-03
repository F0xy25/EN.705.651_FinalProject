# pip install langchain langgraph --quiet
# pip install ipytest
from typing import TypedDict

#Input Schema for the State Graph (Used to ingest user data and influence the state graph)
class UserState(TypedDict):
    User_id: int              #Unique User ID
    preferences: dict         #User preferences (genre, temperature, light_intensity, volume, etc.)
    metadata: dict            #Any other information (Location, Entry Time, ,etc.)

#State Schema to be used for any environmental state updates
class BuildingEventState(TypedDict):
    temperature: float        #Temperature in Farenheit (60-85)
    light_intensity: int      #Light intensity in lumens (0-1000)
    volume: int               #In Decibels (70-120)
    genre: list               #List of genres (pop, rock, jazz, classical, hiphop, country, etc.)
    location: str

class State(TypedDict):
  building_event_state: BuildingEventState

#State Schema for Security Bot
class SecurityBotState(TypedDict):
    robot_id: int              #Unique Robot ID
    metadata: dict             #Any other information (Location,Observations,etc.)

import random
from enum import Enum
# Set in agent after prediction is retrieved if applicable
class Adaptation(Enum):
  UP = 1
  DOWN = 2

# Identifiers which should be referenced in the schema (or the code adapted if they are adapted)
class Purpose(Enum):
  SECURITY = 1
  SOCIAL = 2

class EventLocation(Enum):
  RECEPTION = 1
  HALL = 2
  GARDEN = 3
  DANCE_HALL = 4

class LocationArea(Enum):
  NORTH = 1
  EAST = 2
  SOUTH = 3
  WEST = 4

# Helper methods
def get_optimal_range(state: State, key, subkey=None):
  if subkey is not None:
    return state['min_optimum'][key][subkey], state['max_optimum'][key][subkey]
  else:
    return state['min_optimum'][key], state['max_optimum'][key]


def get_random_val_within_range(state: State, state_param, subkey=None):
  predicted_min_optimum, predicted_max_optimum = get_optimal_range(state, state_param) if subkey is None else get_optimal_range(state, state_param, subkey)
  return random.randrange(int(predicted_min_optimum), int(predicted_max_optimum) + 1)


def switch_location_randomly(existing_location):
  location_options = [location.name for location in EventLocation if location != existing_location]
  return random.choice(location_options)


# Utilities which can changed based on the model's prediction to update an environmental variable
def update_temp(state: State, initialize=False):
  state_param = 'temperature'
  state_update = state["target_value"] if not initialize else get_random_val_within_range(state, state_param)
  if state_update is not None:
    state['building_event_state'].update({state_param: state_update})


def update_lights_lux(state: State, initialize=False):
  # Lights are associated with a location + in the range: 100 - 1000 lux
  state_param = 'light_intensity'
  state_update = state["target_value"] if not initialize else get_random_val_within_range(state, state_param)
  if state_update is not None:
    state['building_event_state'].update({state_param: state_update})


def change_music_volume(state: State, initialize=False):
  state_param = 'volume'
  state_update = state["target_value"] if not initialize else get_random_val_within_range(state, state_param)
  if state_update is not None:
    state['building_event_state'].update({state_param: state_update})


def update_room_location(state: State, initialize=False):
  if initialize:
    state['building_event_state'].update({"location": EventLocation.RECEPTION.name})
  else:
    state['building_event_state'].update({"location": switch_location_randomly(state['location'])})
  # set a default for this
  update_lights_lux(state, initialize=True)
  # dim other room lights
  other_locations = [location.name for location in EventLocation if location.name != state['building_event_state']['location']]
  for _ in other_locations:
    state['building_event_state'].update({'light_intensity': 50})


def make_announcement(state: State, initialize=False):
  announcement = "Welcome to the event!" if initialize else state["target_value"]
  state['building_event_state']['building_event_state'].update({"announcement": announcement})


def ff_genre(state: State, initialize=False):
  if initialize or len(state['building_event_state']['genre']) == 1:
    # set up playlist
    state['building_event_state'].update({"genre": ['genre_1', 'genre_2', 'genre_3']})
  else:
    state['building_event_state'].update({"genre": state['building_event_state']['genre'][1:]})

