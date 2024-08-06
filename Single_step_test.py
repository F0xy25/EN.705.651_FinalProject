from typing import TypedDict, List, Tuple
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, MessagesState
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, conlist
from input_schema_and_utility import get_random_val_within_range, switch_location_randomly, EventLocation
from langchain.output_parsers import JsonOutputToolsParser, ResponseSchema
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
import random
import os

# from langgraph.graph import StateGraph
# ========================================================================
# State
class BuildingEventState(TypedDict):
    temperature: float  # Temperature in Farenheit (60-85)
    light_intensity: int  # Light intensity in lumens (0-1000)
    volume: int  # In Decibels (70-120)
    genres: list  # List of genres (pop, rock, jazz, classical, hiphop, country, etc.)
    location: str # Location of the event in the building
    announcement: str

    @classmethod
    def with_defaults(cls, genres: list, temperature: float = 70.0, light_intensity: int = 55,
                      volume: int = 7, location=EventLocation.RECEPTION.name, announcement='') -> "BuildingEventState":
        return cls(
            genres=genres,
            temperature=temperature,
            light_intensity=light_intensity,
            volume=volume,
            location=location,
            announcement=announcement
        )

    @classmethod
    def with_defaults_low_temp(cls, genres: list, temp: float = 30) -> "BuildingEventState":
        return cls.with_defaults(genres=genres, temperature=temp)

    @classmethod
    def with_defaults_low_light(cls, genres: list, light: float = 10) -> "BuildingEventState":
        return cls.with_defaults(genres=genres, light_intensity=light)

    @classmethod
    def with_defaults_location(cls, genres: list, location: EventLocation = EventLocation.GARDEN) -> "BuildingEventState":
        return cls.with_defaults(genres=genres, location=location.name)


class PredictionState(TypedDict):
    function_name: str  # Name of the function to be called
    target_value: float  # The value to call the chosen function with
    @classmethod
    def with_defaults(cls, function_name: str = '', target_val: str = '') -> "PredictionState":
        return cls(
            function_name=function_name,
            target_value=target_val,
        )


# This is currently a duplicate of Building Event States but eventually it will
# become a signal source for default min + max optimal values for the BES.
class GroupPreferences(TypedDict):
    temperature: float
    light_intensity: int
    volume: int
    genres: list
    location: str
    current_sentiment: str

    @classmethod
    def with_defaults_low(cls, genres: list, temperature: float = 70.0, light_intensity: int = 50,
                      volume: int = 7,  location = EventLocation.RECEPTION.name, current_sentiment = "doing so well") -> "GroupPreferences":
        return cls(
            genres=genres,
            temperature=temperature,
            light_intensity=light_intensity,
            volume=volume,
            location=location,
            current_sentiment=current_sentiment,
        )

    @classmethod
    def with_defaults_high(cls, genres: list, temperature: float = 80.0, light_intensity: int = 60,
                          volume: int = 9, location=EventLocation.DANCE_HALL.name,
                          current_sentiment="Bro this is the best event. I'm obsessed.") -> "GroupPreferences":
        return cls(
            genres=genres,
            temperature=temperature,
            light_intensity=light_intensity,
            volume=volume,
            location=location,
            current_sentiment=current_sentiment,
        )


class OptimalRanges(TypedDict):
    min_optimum: BuildingEventState
    max_optimum: BuildingEventState


class State(TypedDict):
    #Input Schema for the State Graph (Used to ingest user data and influence the state graph)
    User_id: int                   #Unique User ID
    group_preferences: GroupPreferences #User preferences (genre, temperature, light_intensity, volume, etc.)
    metadata: dict                 #Any other information (Location, Entry Time, ,etc.)

    #State Schema to be used for any building event state updates (ENVIRONMENT_VALUES)
    building_event_state: BuildingEventState

    #State Schema for Security Bot
    robot_id: int                  #Unique Robot ID
    metadata: dict                 #Any other information (Location,Observations,etc.)

    # Agent Workflow State Variables
    event_duration_iterator: int   # Used to simulate if the event is over and to the end system loop or not
    current_sentiment: str         # Output by Node 2 User Sentiment Simulation Node
    guests_happy: bool             # Sentiment Analysis output by Node 3 Sentiment Analysis Node to route the output to either Node 1 or 4. True is Happy, False is Sad.
    all_functions: dict            # All the functions that can be used
    predictions: PredictionState
    prior_predictions: Tuple[List[PredictionState], BuildingEventState]
    optimal_ranges: OptimalRanges
    initialized: bool

    # Output details
    messages: str

os.environ['OPENAI_API_KEY'] = ""
workflow = StateGraph(State)

# ========================================================================
# Tools

def update_temp(state: State, initialize=False):
  state_param = 'temperature'
  preset = state['building_event_state']['temperature']
  if initialize and not preset:
      state['building_event_state'].update({state_param: get_random_val_within_range(state, state_param)})
  state_update = 'target_value' in state['predictions']
  if state_update and not initialize:
    state['building_event_state'].update({state_param: state_update})


def update_lights_lux(state: State, initialize=False):
  # Lights are associated with a location + in the range: 100 - 1000 lux
  state_param = 'light_intensity'
  preset = state['building_event_state']['light_intensity']
  if initialize and not preset:
    state['building_event_state'].update({state_param: get_random_val_within_range(state, state_param)})
  elif not initialize:
    state['building_event_state'].update({state_param: state['predictions'].get('target_value')})



def change_music_volume(state: State, initialize=False):
  state_param = 'volume'
  preset = state['building_event_state']['volume']
  if initialize and not preset:
      get_random_val_within_range(state, state_param)
  elif 'target_value' in state['predictions'] and not initialize:
    state['building_event_state'].update({state_param: state['predictions'].get('target_value')})


def update_room_location(state: State, initialize=False):
  preset = state['building_event_state']['location']
  if initialize and not preset:
    state['building_event_state'].update({"location": EventLocation.RECEPTION.name})
  elif not initialize:
    state['building_event_state'].update({"location": switch_location_randomly(state['building_event_state']['location'])})
  # set a default for this
  update_lights_lux(state, initialize=True)


def make_announcement(state: State, initialize=False):
  preset = 'announcement' in state['building_event_state']
  announcement = "Welcome to the event!" if initialize and not preset else state['predictions']["target_value"]
  state['building_event_state'].update({"announcement": announcement})


def ff_genre(state: State, initialize=False):
  current_genre = state['building_event_state']['genres']
  preset = 'genres' in state['building_event_state']
  if initialize and not preset or len(state['building_event_state']['genres']) == 1:
    # set up playlist
    state['building_event_state'].update({"genres": ['piano', 'electronic', 'alternative', 'symphonic']})
  elif not initialize and 'target_value' in state['predictions']:
    state['building_event_state'].update({"genres": state['predictions'].get('target_value')})
  elif not initialize:
    state['building_event_state'].update({"genres": current_genre[1:]})


tools = {
    "update_temp": update_temp,
    "update_lights_lux": update_lights_lux,
    "change_music_volume": change_music_volume,
    "update_room_location": update_room_location,
    "make_announcement": make_announcement,
    "ff_genre": ff_genre
}

def initialize_system_state(state: State, sentiment: str = ''):
    # Agent workflow variables
    # llm agent-communication variables
    state['guests_happy'] = False
    state['current_sentiment'] = sentiment
    # Prediction defaults
    state['building_event_state'] = state['building_event_state'] = BuildingEventState.with_defaults(
        genres=['piano', 'electronic', 'alternative', 'symphonic'],
        temperature=30,
        light_intensity=150,
        volume=1,
        location=EventLocation.GARDEN
    )
    state["prior_predictions"] = []
    state['predictions'] = PredictionState.with_defaults()
    # System variables
    state['event_duration_iterator'] = 0
    state["messages"] = []
    return state

class Node4OutputSchema(BaseModel):
    """
    Output your decision in the following format:
   <function_name>Name of the selected function to be invoked</function_name>
   <target_value>The value to set the environmental factor to</target_value>
    """
    function_name: str
    target_value: str


# 4.(LLM)    Environment Updater Node:
def call_node_4(state):
    #print("NODE 4 ENTERED")
    prompt_template = """
    You are an AI agent working for an Event and Hospitality business. Your role is to analyze the current environment, compare it to optimal ranges, assess event-goer sentiment, and make adjustments to improve the event experience. Follow these instructions carefully:

    1. You will receive four input variables:

    <environment_values>
    {ENVIRONMENT_VALUES}
    </environment_values>

    This contains the current values of various environmental factors at the event.

    <optimal_ranges>
    {OPTIMAL_RANGES}
    </optimal_ranges>

    This specifies the ideal ranges for each environmental factor to ensure event-goer happiness.

    <current_sentiment>
    {CURRENT_SENTIMENT}
    </current_sentiment>

    This represents the current sentiment of event-goers.

    <tools>
    {TOOLS}
    </tools>

    This lists the tools available to you for adjusting environmental factors.

    2. Analyze the current environment by examining the ENVIRONMENT_VALUES.

    3. Compare each value in ENVIRONMENT_VALUES to its corresponding range in OPTIMAL_RANGES. 
    Identify any factors that are outside their optimal ranges.

    4. Assess the CURRENT_SENTIMENT to determine if event-goers are unhappy.
 
    5. If the current sentiment is negative, follow these steps:
    a. Evaluate PRIOR_PREDICTIONS which has a list of tools and values used by them which have been used to try to make 
    them happy, paired with a state. Understand that a tool should not be immediately reused if the environment values 
    haven't changed.
    b. Identify which environmental factor(s) are most likely causing the dissatisfaction 
    c. Determine which factor, if adjusted, would have the most significant positive impact that hasn't been tried, 
    already.
    d. If none seem right just select a factor at random - and hope for the best.
    e. Select the appropriate tool from TOOLS to address this factor.
    f. Decide on the optimal value within the factor's ideal range to set it to.
    g. Respond with a JSON object containing:
    - "function_name": The name of the function to newly execute.  This should be a string.
    - "target_value": The new target value for the function.  This should be a string.
     
    Function Name:
    {function_name}

    Target Value:
    {target_value}
    

    6. If the sentiment is positive or neutral, or if no environmental factors are outside their optimal ranges, output:
    Respond with a JSON object containing:
    - "function_name": pause
    - "target_value": 
    
    Do not include any explanation or reasoning outside of is - your entire output should be the in-character response of the concert-goer.
    Remember, your goal is to improve the event experience by making data-driven decisions based on the provided information."""

    parser = PydanticOutputParser(pydantic_object=Node4OutputSchema)
    PROMPT = PromptTemplate(
        input_variables=["ENVIRONMENT_VALUES", "OPTIMAL_RANGES", "CURRENT_SENTIMENT", "TOOLS", "PRIOR_PREDICTIONS"],
        partial_variables={
            "function_name": lambda: state['building_event_state'].get('function_name', ""),
            "target_value": lambda: state['building_event_state'].get('target_value', ""),
        },
        template=prompt_template
    )

    llm = ChatOpenAI(model="gpt-4o")

    filtered_tools = {}

    for key, value in tools.items():
        if key != state['predictions']['function_name']:
            filtered_tools[key] = value

    prompt = PROMPT.format(
        ENVIRONMENT_VALUES=state.get('building_event_state'),
        OPTIMAL_RANGES=state.get('optimal_ranges'),
        CURRENT_SENTIMENT=state['current_sentiment'],
        PRIOR_PREDICTIONS=state['prior_predictions'],
        TOOLS=filtered_tools,
    )
    msg = llm.invoke(prompt).content
    parsed_output = parser.parse(msg)
    #print("PARSED OuTPUT!")
    #print(parsed_output.function_name)
    #print(parsed_output.target_value)
    if parsed_output.function_name in ['', 'None'] or parsed_output.target_value in ['', 'None']:
        return state

    if (parsed_output.function_name == 'make_announcement'
            or parsed_output.function_name == 'ff_genre'
        or parsed_output.function_name == 'update_room_location'
    ):
        target_val = parsed_output.target_value
    else:
        target_val = int(float(parsed_output.target_value))

    state['predictions'].update({"target_value": target_val})
    state['predictions'].update({"function_name": parsed_output.function_name})
    return parsed_output.function_name



workflow.add_node("Node 4: Environment Updater Node", call_node_4)


#can it just be run without graph
if __name__ == '__main__':

    sentiment_dict = {
        "Dude, I can barely hear myself think! The music's way too loud!": "change_music_volume",
        "Bro these lights are so intense.": "update_lights_lux",
        "Bro I can hardly see you.": "update_lights_lux",
        "Ugh, it's so cold in here.": "update_temp",
        "It's so hot in here.": "update_temp",
        "This music is so boring.": "ff_genre",
        "I'm so tired of this music.": "ff_genre",
        "I don't know, guys... I'm not really feeling this vibe, the energy from this place is just so off right now.": "update_room_location",
        "The room is kinda empty.": "make_announcement",
        "The room is kinda weird.": "update_room_location",
        "The lights are too dim.": "update_lights_lux",
        "The lights are too bright.": "update_lights_lux",
        "The temperature is too low.": "update_temp",
        "The temperature is too high.": "update_temp",
        "I'm tired of this genre.": "ff_genre",
        "The music is too loud.": "change_music_volume",
        "The music is too soft.": "change_music_volume",
        "The room is too crowded.": "update_room_location",
        "The room is too small.": "update_room_location",
        "The music is too intense.": "change_music_volume",
        "The lights are too intense.": "update_lights_lux",
        "It's too cold in here.": "update_temp",
        "The music is too boring.": "ff_genre",
        "I'm not feeling this genre.": "ff_genre",
        "The room is too empty.": "make_announcement",
        "The lights are too dim.": "update_lights_lux",
        "The temperature is too low.": "update_temp",
        "The music is too soft.": "change_music_volume",
        "The room is too crowded.": "update_room_location",
        "The music is too loud.": "change_music_volume",
        "The room is too small.": "update_room_location",
    }

    passed_tests = 0

    for each in sentiment_dict.keys():

        state = State()
        
        t_state = initialize_system_state(state, sentiment=each)
        #print(state)
        #print(t_state)
        fin_state = call_node_4(state)

        print(each,fin_state, sentiment_dict[each])

        if fin_state == sentiment_dict[each]:
            passed_tests += 1

    print(f"Passed {passed_tests} out of {len(sentiment_dict)} tests.")
