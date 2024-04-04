# SQL Agent
## Introduction
SQL Agent

## Prerequisites
- Python 3.9 or higher

## Installation

1. Clone this repository.
2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```
3. Place corresponding OpenAI key, Gemini Key, and Ollama url in utils
4. Place database environment in database/constant_db

## Usage
```bash
python api/app.py
```
Example api:
1. Using FastApi & Langserve:
```bash
{
    "input":{
        "input":"input",
        "tool_llm_name":"gpt-3.5-turbo",
        "agent_llm_name":"gpt-3.5-turbo"
    },
    "config":{},
    "kwargs":{}
}
````
2. Using Flask:
```bash
{
    "input_text": "input",
    "tool_llm_name": "gpt-3.5-turbo",
    "agent_llm_name": "gpt-3.5-turbo"
}
```
