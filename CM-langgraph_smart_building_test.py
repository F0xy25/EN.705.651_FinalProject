import os
from my_api_key import my_api_key
os.environ['OPENAI_API_KEY'] = my_api_key
from langgraph_smart_building import *


def test_call_node_1():
    workflow = StateGraph(State)
    workflow.add_node("Node 1: Changing Environment Simulation Node", call_node_1)
    workflow.add_edge(
        "__start__", "Node 1: Changing Environment Simulation Node")

    app = workflow.compile()

    response = app.invoke(
        {"all_functions": tools},
    )
    assert response['initialized'] == True

