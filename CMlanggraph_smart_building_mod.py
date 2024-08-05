import os
from my_api_key import my_api_key

os.environ['OPENAI_API_KEY'] = my_api_key
from typing import TypedDict, List, Tuple
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, MessagesState
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, conlist
from input_schema_and_utility import get_random_val_within_range, switch_location_randomly, EventLocation
from langchain.output_parsers import JsonOutputToolsParser, ResponseSchema
from langchain_core.output_parsers import PydanticOutputParser
import random


# from langgraph.graph import StateGraph
# ========================================================================
# State
class BuildingEventState(TypedDict):
    temperature: float  # Temperature in Farenheit (60-85)
    light_intensity: int  # Light intensity in lumens (0-1000)
    volume: int  # In Decibels (70-120)
    genres: list  # List of genres (pop, rock, jazz, classical, hiphop, country, etc.)
    location: str  # Location of the event in the building
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
    def with_defaults_location(cls, genres: list,
                               location: EventLocation = EventLocation.GARDEN) -> "BuildingEventState":
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
                          volume: int = 7, location=EventLocation.RECEPTION.name,
                          current_sentiment="doing so well") -> "GroupPreferences":
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


class User(TypedDict):
    User_id: int  # Unique User ID
    current_sentiment: str  # Output by Node 2 User Sentiment Simulation Node
    optimal_ranges: OptimalRanges


class State(TypedDict):
    All_Users: List[User]
    # Input Schema for the State Graph (Used to ingest user data and influence the state graph)
    group_preferences: GroupPreferences  # User preferences (genre, temperature, light_intensity, volume, etc.)
    metadata: dict  # Any other information (Location, Entry Time, ,etc.)

    # State Schema to be used for any building event state updates (ENVIRONMENT_VALUES)
    building_event_state: BuildingEventState

    # State Schema for Security Bot
    robot_id: int  # Unique Robot ID
    metadata: dict  # Any other information (Location,Observations,etc.)

    # Agent Workflow State Variables
    event_duration_iterator: int  # Used to simulate if the event is over and to the end system loop or not

    guests_happy: bool  # Sentiment Analysis output by Node 3 Sentiment Analysis Node to route the output to either Node 1 or 4. True is Happy, False is Sad.
    all_functions: dict  # All the functions that can be used
    predictions: PredictionState
    prior_predictions: Tuple[List[PredictionState], BuildingEventState]
    optimal_ranges: OptimalRanges
    initialized: bool

    # Output details
    messages: str





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
        state['building_event_state'].update(
            {"location": switch_location_randomly(state['building_event_state']['location'])})
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


def initialize_system_state(state: State):
    # Agent workflow variables
    # llm agent-communication variables
    state['guests_happy'] = False

    state['All_Users'] = []
    for i in range(random.randrange(2, 5)):
        state['All_Users'].append(User())

    for user in state['All_Users']:
        user['current_sentiment'] = random.choice(['happy','sad','angry','cold','hot'])
    # Prediction defaults
    state["prior_predictions"] = []
    state['predictions'] = PredictionState.with_defaults()
    # System variables
    state['event_duration_iterator'] = 0
    state["messages"] = []
    return state


def initialize_guest_event_synergy_state(state: State):
    # tests:  set up so that initial event state should line up eventually to group preferences.
    # initial group preferences setting optimal ranges
    state['optimal_ranges'] = OptimalRanges()
    state['optimal_ranges'].update(
        {'min_optimum': GroupPreferences.with_defaults_low(["jazz", "soul"])})
    state['optimal_ranges'].update(
        {'max_optimum': GroupPreferences.with_defaults_high(["jazz", "soul", "hip-hop", "dance"])})

    # Default building state.  The building variables will move to the optimal ranges
    state['building_event_state'] = BuildingEventState.with_defaults(
        genres=['piano', 'electronic', 'alternative', 'symphonic'],
        temperature=30,
        light_intensity=150,
        volume=1,
        location=EventLocation.GARDEN
    )
    # state['building_event_state'] = BuildingEventState.with_defaults_low_temp(genres=["tiny-bop-pop", "baby-shark"])
    # state['building_event_state'] = BuildingEventState.with_defaults_low_light(genres=["jazz", "soul"])
    # state['building_event_state'] = BuildingEventState.with_defaults_location(genres=["jazz", "soul"])

    # Initialize building into the required ranges
    for function in state["all_functions"].values():
        function(state, initialize=True)
    return state


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
    guests_happy: str


class Node4OutputSchema(BaseModel):
    """
    Output your decision in the following format:
   <function_name>Name of the selected function to be invoked</function_name>
   <target_value>The value to set the environmental factor to</target_value>
    """
    function_name: str
    target_value: str


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
        state = initialize_system_state(state)
        state = initialize_guest_event_synergy_state(state)
    state.update({"initialized": True})

    # randomly choose a function from all_functions (a dictionary with the function name as a string as key, and the function itself as a value)
    # that randomizes the value of a certain state variable update the state with that new value
    all_functions = state["all_functions"]
    random_function = random.choice(list(all_functions.values()))
    # TODO: This isn't returning the state + needs updating.
    # state = random_function(state)

    event_duration_iterator = state.get('event_duration_iterator')  # to simulate how long the event is
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
    You can use the current_sentiment range preferences but those are gauges for preferred ranges
    not for the current state.

    Compare the current environment values with your optimal ranges. 
    If even a single value falls outside its optimal range, you must express unhappiness. 
    If all values are within their optimal ranges, express happiness and enjoyment of the event.

    The key environment variables which are readily evaluated numerically are:
    a. temperature
    b. light intensity
    c. volume

    The key environment variables which need some correlation association are:
    a. genres (music genres) - is the value within the mood set by the range within preferred genres?
    The mood can be determined by the emotion the music might invoke and the energy is might inspire.
    b. location - Is the location one of the preferred locations in the min and max optimums?
    It can be assumed that adapting these affects the preference of the variables which are
    numerically based within their optimal ranges.


    3. Respond with a valid JSON object containing:
    - "current_sentiment": Text about how you feel about the event.  This should be a string.

    When responding, use an informal, colloquial tone typical of a concert-goer talking to friends. Your language should be casual, potentially including slang or mild exclamations.

    If unhappy, you may either:
    a) Directly reference the reason for your unhappiness (e.g., "Ugh, it's way too hot in here!") 
    Make sure the statement really reflects the environment variable value relative to the ideal value.
    b) Express general discontent without specifying the cause (e.g., "This isn't as fun as I thought it would be...")
    It's better to add a bit of detail so the friends know what you're talking about.

    If happy, express your enjoyment of the event enthusiastically (e.g., "This is amazing! I'm having the time of my life!")

    Examples of possible responses for a valid JSON output object:
    - "current_sentiment":  "Dude, I can barely hear myself think! The music's way too loud!" (this is unhappy and it is specific to volume)
    - "current_sentiment":  "Bro these lights are so intense." (this is unhappy and it is specific light intensity)
    - "current_sentiment":  "I don't know, guys... I'm not really feeling this vibe." (this is unhappy and maybe multiple things are off)
    - "current_sentiment":  Happy: "Best. Night. Ever! Everything's just perfect!" (this is happy though it isn't clear why)
    - "current_sentiment":  Happy: "Best. Night. Ever! I love ths song!" (this is happy and it is specific to the genre)
    - "current_sentiment":  What a blast! The place has a really warm vibe and I love the flowers!" (this is happy and it is specific to the location)

    Do not include any explanation or reasoning outside of is - your entire output should be the in-character response of the concert-goer.

     """
    parser = PydanticOutputParser(pydantic_object=Node2OutputSchema)

    for user in state['All_Users']:
        PROMPT = PromptTemplate(
            input_variables=["ENVIRONMENT_VALUES", "OPTIMAL_RANGES"],
            partial_variables={
                "current_sentiment": lambda: state.get('guests_happy', ""),
            },
            template=prompt_template
        )

        llm = ChatOpenAI(model="gpt-4o")
        prompt = PROMPT.format(
            ENVIRONMENT_VALUES=state.get('building_event_state'),
            OPTIMAL_RANGES=user.get('optimal_ranges')
        )

        msg = llm.invoke(prompt).content
        parsed_output = parser.parse(msg)
        print('CURRENT SENTIMENT prediction: ' + parsed_output.current_sentiment)
        user.update({"guests_happy": parsed_output.current_sentiment})
    # print("input state action prediction: ")
    # print(state['building_event_state'])
    return state


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

    Examples of positive sentiments might include phrases like "had a great time," "loved the event," "amazing experience," or "can't wait for the next one."
    An example of the response in this case as part of a valid JSON object:
    - "guests_happy": 'true'

    Examples of negative sentiments might include phrases like "disappointed," "waste of time," "poor organization," or "wouldn't recommend."
    An example of the response in this case as part of a valid JSON object:
    - "guests_happy": 'false'

    5. Respond with a valid JSON object containing:
    - "guests_happy": The state of the guests happiness which can be true or false.  This should be a string.

    Remember to focus solely on determining whether the sentiment indicates happiness or sadness in the analysis. 
    Do not consider other emotions or nuances beyond this binary classification.

    Provide your analysis and reasoning in the "reasoning" variable of the JSON object."""

    parser = PydanticOutputParser(pydantic_object=Node3OutputSchema)
    PROMPT = PromptTemplate(
        input_variables=["current_sentiment"],
        output_variables=["guests_happy"],
        partial_variables={
            "guests_happy": lambda: state['building_event_state'].get('guests_happy', ""),
        },
        template=prompt_template
    )

    llm = ChatOpenAI(model="gpt-4o")

    filtered_tools = {}

    for key, value in tools.items():
        if key != state['predictions']['function_name']:
            filtered_tools[key] = value

    pass

    prompts = [PROMPT.format(
        current_sentiment=user["current_sentiment"]) for user in state['All_Users']]

    msgs = [llm.invoke(prompt).content for prompt in prompts]
    parsed_outputs = [parser.parse(msg) for msg in msgs]
    print("TEST tool and value: ")



    # print("input state action prediction: ")
    # print(state['building_event_state'])
    for i in range(len(parsed_outputs)):
        state['All_Users'][i].update({"current_sentiment": parsed_outputs[i].guests_happy})

    print([parsed_output for parsed_output in parsed_outputs])

    #state.update({"guests_happy": parsed_output.guests_happy})

    #aggregate_guest_happiness(parsed_outputs,state)
    check_key_guest_happiness(parsed_outputs,state)

    pass
    return state


def check_key_guest_happiness(parsed_outputs,state):
    if parsed_outputs[0].guests_happy == 'true':
        state.update({"guests_happy": True})
        print("Guest Satisfied")
    else:
        state.update({"guests_happy": False})


def aggregate_guest_happiness(parsed_outputs,state):
    total_guests = len(parsed_outputs)
    THRESHHOLD = .6
    count = 0
    for parsed_output in parsed_outputs:
        if parsed_output.guests_happy == 'true':
            count = count + 1
    if count/total_guests > THRESHHOLD:
        state.update({"guests_happy": True})
        print("Guests Satisfied")
    else:
        state.update({"guests_happy": False})
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


    #Do we want individual prompts for each guest's current_sentiment?
    prompt = PROMPT.format(
        ENVIRONMENT_VALUES=state.get('building_event_state'),
        OPTIMAL_RANGES=state['All_Users'][0].get('optimal_ranges'),#key guest is guest 0
        CURRENT_SENTIMENT=state['guests_happy'],# changed from current_sentiment to guests_happy
        PRIOR_PREDICTIONS=state['prior_predictions'],
        TOOLS=filtered_tools,
    )
    msg = llm.invoke(prompt).content
    parsed_output = parser.parse(msg)
    print("PARSED OuTPUT!")
    print(parsed_output.function_name)
    print(parsed_output.target_value)
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
    return state


# 5.(TOOL)  Tool Node:
def call_node_5(state):
    function_name = state["predictions"]["function_name"]
    all_functions = state["all_functions"]
    if function_name in all_functions:
        chosen_function = all_functions[function_name]
        state['prior_predictions'] += (state["predictions"], state['building_event_state'])
        state = chosen_function(state)
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
    "Node 1: Changing Environment Simulation Node", is_event_done_yet_condition,
    {"Node 2: User Sentiment Simulation Node": "Node 2: User Sentiment Simulation Node", "FINISH": END})
workflow.add_edge(
    "Node 2: User Sentiment Simulation Node", "Node 3: Sentiment Analysis Node")
workflow.add_conditional_edges(
    "Node 3: Sentiment Analysis Node", guest_sentiment_condition,
    {"Node 1: Changing Environment Simulation Node": "Node 1: Changing Environment Simulation Node",
     "Node 4: Environment Updater Node": "Node 4: Environment Updater Node"})
workflow.add_edge(
    "Node 4: Environment Updater Node", "Node 5: Tool Node")
workflow.add_edge(
    "Node 5: Tool Node", "Node 2: User Sentiment Simulation Node")

# ========================================================================
# Run Graph
app = workflow.compile()

response = app.invoke(
    {"all_functions": tools},
)

for message in response["messages"]:
    string_representation = f"{message.type.upper()}: {message.content}\n"
    print(string_representation)
