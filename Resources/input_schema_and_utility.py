# pip install langchain langgraph --quiet
# pip install ipytest
from typing import TypedDict

#Input Schema for the State Graph (Used to ingest user data and influence the state graph)
class UserState(TypedDict):
    User_id: int              #Unique User ID
    preferences: dict         #User preferences (genre, temperature, light_intensity, volume, etc.)
    metadata: dict            #Any other information (Location, Entry Time, ,etc.)

#State Schema to be used for any environmental state updates
class EnvironmentState(TypedDict):
    temperature: float        #Temperature in Farenheit (60-85)
    light_intensity: int      #Light intensity in lumens (0-1000)
    volume: int               #In Decibels (70-120)
    genre: list               #List of genres (pop, rock, jazz, classical, hiphop, country, etc.)

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
def get_optimal_range(state: EnvironmentState, key, subkey=None):
  if subkey is not None:
    return state['min_optimum'][key][subkey], state['max_optimum'][key][subkey]
  else:
    return state['min_optimum'][key], state['max_optimum'][key]


def find_adaptation_from_optimal_range(state: EnvironmentState, optimal_min, optimal_max, current):
  adaptation = state['prediction_detail']['adaptation']
  if adaptation == Adaptation.UP:
    return random.randrange(current, optimal_max + 1)
  elif adaptation == Adaptation.DOWN:
    return random.randrange(optimal_min, current)
  return None


def get_min_max_update(state: EnvironmentState, state_param, subkey=None):
  predicted_min_optimum, predicted_max_optimum = get_optimal_range(state, state_param) if subkey is None else get_optimal_range(state, state_param, subkey)
  current = state[state_param] if subkey is None else state[state_param][subkey]
  return find_adaptation_from_optimal_range(state, predicted_min_optimum, predicted_max_optimum, current)


def switch_location_randomly(existing_location):
  location_options = [location.name for location in EventLocation if location != existing_location]
  return random.choice(location_options)


# Utilities which can changed based on the model's prediction to update an environmental variable
def update_temp(state: EnvironmentState):
  state_param = 'temperature'
  state_update = get_min_max_update(state, state_param)
  if state_update is not None:
    state[state_param] = state_update


def update_lights_lux(state: EnvironmentState):
  # Lights are associated with a location + in the range: 100 - 1000 lux
  state_param = 'lights'
  current_location = state['room_location']
  state_update = get_min_max_update(state, state_param, current_location)
  if state_update is not None:
    state[state_param][current_location] = state_update


def change_music_volume(state: EnvironmentState):
  state_param = 'music_volume'
  state_update = get_min_max_update(state, state_param)
  if state_update is not None:
    state[state_param] = state_update


def update_room_location(state: EnvironmentState):
  state['room_location'] = switch_location_randomly(state['room_location'])
  test = update_lights_lux(state)
  # set a default for this
  update_lights_lux(state)
  # dim other room lights

  other_locations = [location.name for location in EventLocation if location.name != state['room_location']]
  for location in other_locations:
    state['lights'][location] = 50


def make_announcement(state: EnvironmentState):
  state['announcement'] = state['prediction_detail']['language_update']


def skip_song(state: EnvironmentState):
  # assumed that this is a skip, but it could include more complex requests
  state['music_playlist'] = state['music_playlist'][1:]


def generate_report_from_undercover_security_bot(state: SecurityBotState):
  # The bot cruises around and evaluates the level of chaos at the event location
  # These updates are iterative not random for orchestrating escalation + de-escalation
  current_area = state['bot']['area'].name
  adaptation = state['prediction_detail']['adaptation']
  if adaptation == Adaptation.UP:
    state[current_area]['chaos'] = state[current_area]['chaos'] + 1
  elif adaptation == Adaptation.DOWN:
    state[current_area]['chaos'] = state[current_area]['chaos'] - 1

  # The bot moves if the chaos is below the maximum threshold
  if state[current_area]['chaos'] < state['max_optimum']['chaos']:
    state['bot']['area'] = random.choice([area for area in LocationArea if area.name != current_area])



# The bot goes to certain areas and delivers security messages in times of chaos
def purpose_undercover_security_bot(state: SecurityBotState):
  predicted_min_optimum = state['min_optimum']['chaos']
  predicted_max_optimum = state['max_optimum']['chaos']
  # Below the range of normal 'chaos' requires a security response as well.
  prior_objective = state['bot']['objective']
  if state['chaos'] > predicted_max_optimum or state['chaos'] < predicted_min_optimum:
    state['bot']['objective'] = Purpose.SECURITY
    state['bot']['decibal'] = state['music_volume'] + 50
    state['music_volume'] = state['music_volume'] - 50
  else:
    state['bot']['objective'] = Purpose.SOCIAL
    if prior_objective == Purpose.SECURITY:
      state['music_volume'] = state['music_volume'] + 50
      state['bot']['decibal'] = state['music_volume'] - 50
  state['bot']['message'] = state['prediction_detail']['language_update']