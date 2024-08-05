# EN.705.651_FinalProject: Building Environment modifier with LangGraph


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Links](#links)
- [Contributors](#contributors)

## Overview
Our project idea is to build a system which works as a software backbone for a LLM Agent which controls a Smart Building.  Output sequences to this system will include process-based tasks such as temperature, lights, ambiance, energy-processes and logistic management.  The overall idea we’re exploring is the software design of the translation layer for sequencing tasks using the LLM integration as the task generator based on inputs sourced from signals in the building and prompts associated with or developed by those building state details. The deliverable is an application where the system inputs are clearly defined, where outputs are clearly expressed, and where an established metric evaluation expresses the effectiveness of the LLM integration objective to correctly determine and sequence tasks.

Rather than actually integrating with a legitimate smart home for testing, which would require an enormous amount of resources, and likely the integration of multiple data pipelines, we will instead generate proxies for this input data as follows:

- Temperature
- Light intensity
- Genre of Music
- location

To develop the LLM agent to be utilized in this system, the team chose to use LangGraph. Below is a visualization of the agentic system. 


![Alt text](/Presentation_Materials/FinalProjectArchitecture.jpg)

1. (NO-LLM) Changing Environment Simulation Node:
    Runs a function that randomly change state variables

2. (LLM) User Sentiment Simulation Node: 
    Based off the {optimal values associated with user id} and the {changes to the intput} (fed in dynamically), its instructed to imagine it is a person and output how it feels (good or bad) depending on if the values are within the optimal range or not.

3. (LLM) Sentiment Analysis Node: 
    Performs Sentiment Analaysis. Takes in output from 2. and determines if sentiment is good or bad. Outputs which node should be next based off if the sentiment is judged to be good or bad.

4. (LLM) Environment Updater Node:
    Takes in the current values of each variable, the sentiment from 2., the analysis from 3., and optimal values for each variable, and then decides which variable needs to be changed
    to improve sentiment of the user, and then outputs a function to be called and the inputs to that function.

5. (TOOL) Tool Node:
    Calls chosen tool decided by 4, and updates the state with that


## Installation
To install the project, follow these steps:

```sh
# Clone the repository
git clone https://github.com/F0xy25/EN.705.651_FinalProject.git

# Navigate to the project directory
cd EN.705.651_FinalProject

# Install dependencies
pip install -r requirements.txt
```


## Usage
Provide instructions and examples for using your project. For example:

Important to note, this code utilizes OpenAI's ChatGPT-4o model. Therefore, it is required that a project key be provided to the program. You can aqcuire your project key by following [OpenAI's instructions]. (https://help.openai.com/en/articles/9186755-managing-your-work-in-the-api-platform-with-projects)

Your project key should be inserted in the header of langgraph_smart_building.py, then the project can be run with the following command:

```sh
# Run the application
python langgraph_smart_building.py
```
Where all output will be printed to the terminal. 


## File Structure
```plaintext
EN.705.651_FinalProject/
├── Presentation_Materials/
│   ├── .gitkeep
|   ├── FinalProjectArchitecture.jpg
├── Project_Deliverables/
│   └── .gitkeep
├── .gitignore
├── CM-langgraph_smart_building.py
├── input_schema_and_utility.py
├── langgraph_smart_building.py
├── requirements.txt
└── README.md
```

Explain the purpose of key directories and files.

## Contributing
If you would like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links
- [Project Repository](https://github.com/your-username/your-repo-name)
- [Live Demo](https://your-live-demo-link.com)
- [Documentation](https://your-documentation-link.com)
- [System Orchestration](https://docs.google.com/drawings/d/1itQ5FZCUh4hbS-R1wBChX8EU8FWujsHUNesk57L5UnE/edit)
- [Project Proposal](https://docs.google.com/document/d/10s2cT2RUXrkkUDlLlY-RqcHuX136EN6bAoMSd-D0xSQ/edit)

## Contributors:

If you have further questions about this project, please do reach out to the following:

- Gavin Fox
- Catherine Johnson
- Marc Roube
- Charles Mackey
