from input_schema_and_utility import Adaptation, EventLocation, Purpose, LocationArea
from input_schema_and_utility import (
    find_adaptation_from_optimal_range,
    get_min_max_update,
    switch_location_randomly,
    update_room_location,
    skip_song,
    purpose_undercover_security_bot,
    generate_report_from_undercover_security_bot)


def test_find_adaptation_from_optimal_range_up():
    state = {
      'prediction_detail': {
          'adaptation': Adaptation.UP
      }
    }
    assert find_adaptation_from_optimal_range(state, 10, 100, 50) > 50


def test_get_min_max_update_test():
    state = {
      'min_optimum': {'lights': 10},
      'max_optimum': {'lights': 100},
      'prediction_detail': {
          'adaptation': Adaptation.DOWN
      },
      'lights': 34,
    }
    assert get_min_max_update(state, 'lights') < 34


def test_switch_location_randomly():
    assert switch_location_randomly(EventLocation.GARDEN) in [location.name for location in EventLocation if location.name != EventLocation.GARDEN.name]


default_state = {
    'room_location': EventLocation.DANCE_HALL,
    'lights': {
        EventLocation.DANCE_HALL.name: 25,
        EventLocation.HALL.name: 25,
        EventLocation.GARDEN.name: 25,
        EventLocation.RECEPTION.name: 25,
        },
    'min_optimum': {'lights': {
        EventLocation.DANCE_HALL.name: 10,
        EventLocation.HALL.name: 10,
        EventLocation.GARDEN.name: 10,
        EventLocation.RECEPTION.name: 10,
        }},
    'max_optimum': {'lights': {
        EventLocation.DANCE_HALL.name: 50,
        EventLocation.HALL.name: 50,
        EventLocation.GARDEN.name: 50,
        EventLocation.RECEPTION.name: 50,
        }},
    'prediction_detail': {
        'adaptation': Adaptation.UP},
}


def test_update_room_location(state = default_state):
    orig_location = state['room_location'].name
    update_room_location(state)
    assert state['room_location'] in [location.name for location in EventLocation if location.name != orig_location]
    assert state['lights'][state['room_location']] > state['min_optimum']['lights'][state['room_location']]


def test_skip_song(state = {'music_playlist': ['song_1', 'song_2', 'song_3']}):
    assert state['music_playlist'][0] == 'song_1'
    skip_song(state)
    assert state['music_playlist'][0] == 'song_2'


def test_generate_report_from_undercover_security_bot(state = {
    'bot': {
      'area': LocationArea.NORTH,
      'objective': Purpose.SOCIAL,
      'decibal': 100
    },
    'prediction_detail': {
      'language_update': 'Security message',
      'adaptation': Adaptation.UP
    },
    'max_optimum': {'chaos': 100},
    'NORTH': {
      'chaos': 10
    }
}):
    generate_report_from_undercover_security_bot(state)
    assert state['NORTH']['chaos'] == 11
    assert state['bot']['area'] in [area for area in LocationArea if area.name != LocationArea.NORTH]


def test_purpose_undercover_security_bot_responds_to_chaos(state = {
    'min_optimum': {'chaos': 10},
    'max_optimum': {'chaos': 100},
    'music_volume': 50,
    'chaos': 101,
    'bot': {
      'area': LocationArea.NORTH,
      'objective': Purpose.SOCIAL,
      'decibal': 100
    },
    'prediction_detail': {
      'language_update': 'Please Lower Chaos',
      'adaptation': Adaptation.UP
    }
}):
    purpose_undercover_security_bot(state)
    assert state['bot']['objective'] == Purpose.SECURITY
    assert state['music_volume'] == 0
    assert state['bot']['decibal'] == 100
    assert state['bot']['message'] == 'Please Lower Chaos'


def test_purpose_undercover_security_bot_no_response(state = {
    'min_optimum': {'chaos': 10},
    'max_optimum': {'chaos': 100},
    'music_volume': 50,
    'chaos': 99,
    'bot': {
      'area': LocationArea.NORTH,
      'objective': Purpose.SECURITY,
      'decibal': 100
    },
    'prediction_detail': {
      'language_update': 'Enjoy Event',
      'adaptation': Adaptation.UP
    }
}):
    purpose_undercover_security_bot(state)
    assert state['bot']['objective'] == Purpose.SOCIAL
    assert state['music_volume'] == 100
    assert state['bot']['decibal'] == 50
    assert state['bot']['message'] == 'Enjoy Event'