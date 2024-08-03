from input_schema_and_utility import Adaptation, EventLocation, Purpose, LocationArea
from input_schema_and_utility import (
    update_lights_lux,
    switch_location_randomly,
    update_room_location,
    ff_genre)

default_state_lights = {
  'min_optimum': {'light_intensity': 10},
  'max_optimum': {'light_intensity': 100},
  'building_event_state': {
    'light_intensity': 34
  },
  'target_value': 29
}
def test_update_lights():
    update_lights_lux(default_state_lights)
    assert default_state_lights['building_event_state']['light_intensity'] == 29

def test_update_lights_init():
    update_lights_lux(default_state_lights, initialize=True)
    assert default_state_lights['building_event_state']['light_intensity'] <= default_state_lights['max_optimum']['light_intensity']
    assert default_state_lights['building_event_state']['light_intensity'] >= default_state_lights['min_optimum']['light_intensity']


def test_switch_location_randomly():
    assert switch_location_randomly(EventLocation.GARDEN) in [location.name for location in EventLocation if location.name != EventLocation.GARDEN.name]


default_state_rooms = {
    'min_optimum': {'light_intensity': 10},
    'max_optimum': {'light_intensity': 50},
    'building_event_state': {
        'location': EventLocation.DANCE_HALL,
        'light_intensity': {
            EventLocation.DANCE_HALL.name: 25,
            EventLocation.HALL.name: 25,
            EventLocation.GARDEN.name: 25,
            EventLocation.RECEPTION.name: 25,
        }
    },
    'target_value': EventLocation.DANCE_HALL.name
}


def test_update_room_location(state=default_state_rooms):
    orig_location = state['location'].name
    update_room_location(state)
    assert state['building_event_state']['location'] in [location.name for location in EventLocation if location.name != orig_location]
    assert state['building_event_state']['light_intensity'] > state['min_optimum']['light_intensity']

def test_update_room_location_init(state=default_state_rooms):
    orig_location = state['location'].name
    update_room_location(state, initialize=True)
    assert state['building_event_state']['location'] == EventLocation.RECEPTION.name
    assert state['building_event_state']['light_intensity'] > state['min_optimum']['light_intensity']


default_state_music = {
    'building_event_state': {'genre': ['genre_1', 'genre_2', 'genre_3']}
}
def test_ff_genre(state=default_state_music):
    assert state['building_event_state']['genre'][0] == 'genre_1'
    ff_genre(state)
    assert state['building_event_state']['genre'][0] == 'genre_2'
