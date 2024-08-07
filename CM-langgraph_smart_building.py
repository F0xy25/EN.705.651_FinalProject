from typing import TypedDict
import os#moved os to the top
from my_api_key import my_api_key#store secret api key in python file my_api_key.py
if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = my_api_key#need to set api key before importing ChatOpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, MessagesState
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, conlist
from input_schema_and_utility import get_random_val_within_range, switch_location_randomly, EventLocation
import random


# from langgraph.graph import StateGraph
# ========================================================================
# State
class BuildingEventState(TypedDict):
    temperature: float  # Temperature in Farenheit (60-85)
    light_intensity: int  # Light intensity in lumens (0-1000)
    volume: int  # In Decibels (70-120)
    genre: list  # List of genres (pop, rock, jazz, classical, hiphop, country, etc.)
    location: str # Location of the event in the building


# This is currently a duplicate of Building Event States but eventually it will
# become a signal source for default min + max optimal values for the BES.
class GroupPreferences(TypedDict):
    genre: str
    temperature: float
    light_intensity: int
    volume: int
    genre: list
    location: str

    @classmethod
    def with_defaults(cls, genres: list, temperature: float = 70.0, light_intensity: int = 50,
                      volume: int = 5) -> "GroupPreferences":
        return cls(
            genre=genres,
            temperature=temperature,
            light_intensity=light_intensity,
            volume=volume,
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
    function_name: str             # Name of the function to be called
    target_value: float            # The value to call the chosen function with
    optimal_ranges: OptimalRanges
    initialized: bool

    # Output details
    messages: str

#from my_api_key import my_api_key
#os.environ['OPENAI_API_KEY'] = my_api_key#moved to top
workflow = StateGraph(State)


# ========================================================================
# Tools

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
    state['building_event_state'].update({"location": switch_location_randomly(state['building_event_state']['location'])})
  # set a default for this
  update_lights_lux(state, initialize=True)
  # dim other room lights
  other_locations = [location.name for location in EventLocation if location.name != state['building_event_state']['location']]
  for _ in other_locations:
    state['building_event_state'].update({'light_intensity': 50})


def make_announcement(state: State, initialize=False):
  announcement = "Welcome to the event!" if initialize else state["target_value"]
  state['building_event_state'].update({"announcement": announcement})


def ff_genre(state: State, initialize=False):
  if initialize or len(state['building_event_state']['genre']) == 1:
    # set up playlist
    state['building_event_state'].update({"genre": ['genre_1', 'genre_2', 'genre_3']})
  else:
    state['building_event_state'].update({"genre": state['building_event_state']['genre'][1:]})


tools = {
    "update_temp": update_temp,
    "update_lights_lux": update_lights_lux,
    "change_music_volume": change_music_volume,
    "update_room_location": update_room_location,
    "make_announcement": make_announcement,
    "ff_genre": ff_genre
}

# ========================================================================
# Pydantic Output Schemas

class Node2OutputSchema(BaseModel):
    """Given your preferences, a sentiment that is either positive or negative about how you feel right now"""
    current_sentiment: str

class Node3OutputSchema(BaseModel):
    """Your determination as a boolean, whether or not the sentiment presented to you is postive or negative"""
    guests_happy: bool 

class Node4OutputSchema(BaseModel):
    """
    Output your decision in the following format:
   <function_name>Name of the selected function to be invoked</function_name>
   <target_value>The value to set the environmental factor to</target_value>
    """
    function_name: str
    target_value: float




# ========================================================================
# Nodes
# EXPLANATION #
    # 1.(NO-LLM) Changing Environment Simulation Node:
    #            Runs a function that randomly change state variables

    # 2.(LLM)    User Sentiment Simulation Node: 
    #            Based off the {optimal values associated with user id} and the {changes to the intput} 
    #            (fed in dynamically), its instructed to imagine it is a person and output how it feels (good or bad) 
    #            depending on if the values are within the optimal range or not.

    # 3.(LLM)    Sentiment Analysis Node: 
    #            Performs Sentiment Analaysis. Takes in output from 2. and determines if sentiment is good or bad.
    #            Outputs which node should be next based off if the sentiment is judged to be good or bad.
    
    # 4.(LLM)    Environment Updater Node:
    #            Takes in the current values of each variable, the sentiment from 2., the analysis from 3., 
    #            and optimal values for each variable, and then decides which variable needs to be changed
    #            to improve sentiment of the user, and then outputs a function to be called and the inputs 
    #            to that function.

    # 5.(TOOL)  Tool Node:
    #           Calls chosen tool decided by 4, and updates the state with that

    # 6. FINISH


# 1.(NO-LLM) Changing Environment Simulation Node: 
def call_node_1(state):
    # initialize everything if the state is not been set.
    # eventually min optimum + max optimum will be in a prediction node.
    if not state["initialized"]:
        all_functions = state["all_functions"]
        state['building_event_state'] = BuildingEventState()
        state['optimal_ranges'] = OptimalRanges()
        state['optimal_ranges'].update({'min_optimum': GroupPreferences.with_defaults(["jazz", "hip-hop"])})
        state['optimal_ranges'].update({'max_optimum': GroupPreferences.with_defaults(["jazz", "hip-hop"])})
        state['event_duration_iterator'] = 0
        state["messages"] = []
        for key, value in GroupPreferences.with_defaults(["jazz", "hip-hop"]).items():
            state['optimal_ranges']['min_optimum'].update({key: value})
            state['optimal_ranges']['max_optimum'].update({key: value})

        # Set all building event variables from the optimal ranges
        for function in all_functions.values():
            function(state, initialize=True)
    state.update({"initialized": True})


    # randomly choose a function from all_functions (a dictionary with the function name as a string as key, and the function itself as a value)
    # that randomizes the value of a certain state variable update the state with that new value
    all_functions = state["all_functions"]
    random_function = random.choice(list(all_functions.values()))
    # TODO: This isn't returning the state + needs updating.
    # state = random_function(state)

    event_duration_iterator = state.get('event_duration_iterator') # to simulate how long the event is
    event_duration_iterator += 1

    state.update({"event_duration_iterator": event_duration_iterator})
    return state



# 2.(LLM)    User Sentiment Simulation Node: 
def call_node_2(state):
    prompt_template = """
    You are an AI agent simulating a human attending an event, specifically a concert. You will be given information about the current environment and your preferences. Your task is to respond as if you were communicating with friends at the concert, expressing either happiness or unhappiness based on the environmental conditions.

    You will receive two sets of information:

    1. Current environment values:
    <environment_values>
    {ENVIRONMENT_VALUES}
    </environment_values>

    2. Your optimal ranges for happiness:
    <optimal_ranges>
    {OPTIMAL_RANGES}
    </optimal_ranges>

    Compare the current environment values with your optimal ranges. If even a single value falls outside its optimal range, you must express unhappiness. If all values are within their optimal ranges, express happiness and enjoyment of the event.

    When responding, use an informal, colloquial tone typical of a concert-goer talking to friends. Your language should be casual, potentially including slang or mild exclamations.

    If unhappy, you may either:
    a) Directly reference the reason for your unhappiness (e.g., "Ugh, it's way too hot in here!")
    b) Express general discontent without specifying the cause (e.g., "This isn't as fun as I thought it would be...")

    If happy, express your enjoyment of the event enthusiastically (e.g., "This is amazing! I'm having the time of my life!")

    Examples of possible responses:
    - Unhappy: "Dude, I can barely hear myself think! The music's way too loud!"
    - Unhappy: "I don't know, guys... I'm not really feeling this vibe."
    - Happy: "Best. Night. Ever! Everything's just perfect!"

    Provide your response within <response> tags. Do not include any explanation or reasoning outside of these tags - your entire output should be the in-character response of the concert-goer.
        """
    PROMPT = PromptTemplate(
        input_variables=["ENVIRONMENT_VALUES", "OPTIMAL_RANGES"], 
        template=prompt_template
    )

    llm = ChatOpenAI(model="gpt-4o")
    prompt = PROMPT.format(ENVIRONMENT_VALUES=state.get('building_event_state'), OPTIMAL_RANGES=state.get('optimal_ranges'))


    return prompt | llm.with_structured_output(Node2OutputSchema)


# 3.(LLM)    Sentiment Analysis Node: 
def call_node_3(state):
    prompt_template = """
    You are a Sentiment Analysis Agent tasked with determining whether an event-goer is happy or sad based on their survey response. You will receive the response in the following variable:

    <current_sentiment>
    {current_sentiment}
    </current_sentiment>

    Your job is to analyze this sentiment and determine if it indicates that the event-goer is happy or sad. Follow these guidelines:

    1. Carefully read the entire sentiment response.
    2. Look for key words and phrases that indicate positive or negative emotions.
    3. Consider the overall tone of the response.
    4. Pay attention to any specific comments about the event experience.

    Based on your analysis, you will determine if the guest is happy or not. If the sentiment analysis indicates that the guest/concert-goer is happy, you will set the 'guests_happy' value to true. If the sentiment analysis indicates that the guest/concert-goer is sad or at least not happy, you will set the 'guests_happy' value to false.

    Output your result in the following format:
    <result>
    {
    "guests_happy": boolean_value
    }
    </result>

    Where boolean_value is either true or false.

    Examples of positive sentiments might include phrases like "had a great time," "loved the event," "amazing experience," or "can't wait for the next one."

    Examples of negative sentiments might include phrases like "disappointed," "waste of time," "poor organization," or "wouldn't recommend."

    Remember to focus solely on determining whether the sentiment indicates happiness or sadness. Do not consider other emotions or nuances beyond this binary classification.

    Provide your analysis and reasoning before giving the final result."""
    
    PROMPT = PromptTemplate(
        input_variables=["current_sentiment"], 
        template=prompt_template
    )

    llm = ChatOpenAI(model="gpt-4o")
    prompt = PROMPT.format(current_sentiment=state["current_sentiment"])

    return prompt | llm.with_structured_output(Node3OutputSchema)


# 4.(LLM)    Environment Updater Node:
def call_node_4(state):
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

    3. Compare each value in ENVIRONMENT_VALUES to its corresponding range in OPTIMAL_RANGES. Identify any factors that are outside their optimal ranges.

    4. Assess the CURRENT_SENTIMENT to determine if event-goers are unhappy.

    5. If the sentiment is negative, follow these steps:
    a. Identify which environmental factor(s) are most likely causing the dissatisfaction.
    b. Determine which factor, if adjusted, would have the most significant positive impact.
    c. Select the appropriate tool from TOOLS to address this factor.
    d. Decide on the optimal value within the factor's ideal range to set it to.

    6. Output your decision in the following format:
    <decision>
    <function_name>Name of the selected tool</function_name>
    <target_value>The value to set the environmental factor to</target_value>
    </decision>

    7. If the sentiment is positive or neutral, or if no environmental factors are outside their optimal ranges, output:
    <decision>No action needed</decision>

    Here's an example of how your output might look:

    <decision>
    <function_name>adjust_temperature</function_name>
    <target_value>72</target_value>
    </decision>

    Remember, your goal is to improve the event experience by making data-driven decisions based on the provided information."""
    

    PROMPT = PromptTemplate(
        input_variables=["ENVIRONMENT_VALUES", "OPTIMAL_RANGES", "CURRENT_SENTIMENT", "TOOLS"], 
        template=prompt_template
    )

    prompt = PROMPT.format(ENVIRONMENT_VALUES=state.get('building_event_state'), OPTIMAL_RANGES=state.get('optimal_ranges'), CURRENT_SENTIMENT=state['guests_happy'], TOOLS=tools)

    llm = ChatOpenAI(model="gpt-4o")
    
    return prompt | llm.with_structured_output(Node4OutputSchema)


# 5.(TOOL)  Tool Node:
def call_node_5(state):
    chosen_function = state["all_functions"][state["function_name"]]
    state = chosen_function(state["target_value"])
    return state
   

workflow.add_node("Node 1: Changing Environment Simulation Node", call_node_1)
workflow.add_node("Node 2: User Sentiment Simulation Node", call_node_2)
workflow.add_node("Node 3: Sentiment Analysis Node", call_node_3)
workflow.add_node("Node 4: Environment Updater Node", call_node_4)
workflow.add_node("Node 5: Tool Node", call_node_5)




# ========================================================================
# Edges 
# EXPLANATION #
    # Start to Node 1.
    # 1. conditional edge to 2.
        # (if event is still going)
    # 1. conditional edge to 6. FINISH
        # (if event is over perhaps based off an iterator)
    # 2. unconditional edge to 3.
    # 3. conditional edge to 1.
        # (if sentiment of input from 2 is good)
    # 3. conditional edge to 4.
        # (if sentiment of input from 2 is bad)
    # 4. unconditional edge to 2.



# IMPLEMENTATION #
# DEFINE THE FUNCTIONS FOR CONDITONAL EDGES
# 1. conditional edge to 2. (if event is still going) or 1. to 6. FINISH (if event is over based off an iterator)
def is_event_done_yet_condition(state):
    if state['event_duration_iterator'] < 20:
        return "Node 2: User Sentiment Simulation Node"
    else:
      return "FINISH"

# 3. conditional edge to 1. (if sentiment of input from 2 is good) or 3. conditional edge to 4. (if sentiment of input from 2 is bad)
def guest_sentiment_condition(state):
    if state['guests_happy'] == True:
        return "Node 1: Changing Environment Simulation Node"
    else:
      return "Node 4: Environment Updater Node"


# Define the edges
workflow.add_edge(
    "__start__", "Node 1: Changing Environment Simulation Node")
workflow.add_conditional_edges(
    "Node 1: Changing Environment Simulation Node", is_event_done_yet_condition, {"Node 2: User Sentiment Simulation Node": "Node 2: User Sentiment Simulation Node", "FINISH": END})
workflow.add_edge(
    "Node 2: User Sentiment Simulation Node", "Node 3: Sentiment Analysis Node")
workflow.add_conditional_edges(
    "Node 3: Sentiment Analysis Node", guest_sentiment_condition, {"Node 1: Changing Environment Simulation Node": "Node 1: Changing Environment Simulation Node", "Node 4: Environment Updater Node": "Node 4: Environment Updater Node"})
workflow.add_edge(
    "Node 4: Environment Updater Node", "Node 5: Tool Node")
workflow.add_edge(
    "Node 5: Tool Node", "Node 2: User Sentiment Simulation Node")



# ========================================================================
# Run Graph
if __name__ == '__main__':
    app = workflow.compile()

    response = app.invoke(
        {"all_functions": tools},
    )

    for message in response["messages"]:
        string_representation = f"{message.type.upper()}: {message.content}\n"
        print(string_representation)
