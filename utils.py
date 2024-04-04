from langchain.chat_models import ChatOpenAI

from langchain_community.llms import Ollama

from langchain_google_genai import ChatGoogleGenerativeAI

from constants import chat_openai_model_kwargs, langchain_chat_kwargs

# Optional, set the API key for OpenAI if it's not set in the environment.
# os.environ["OPENAI_API_KEY"] = "xxxxxx"


def get_chat_openai(model_name):
    """
    Returns an instance of the ChatOpenAI class initialized with the specified model name.

    Args:
        model_name (str): The name of the model to use.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class.

    """
    llm = ChatOpenAI(
        openai_api_key="",
        model_name=model_name,
        model_kwargs=chat_openai_model_kwargs,
        **langchain_chat_kwargs
    )
    return llm

def get_ollama_llms(model_name):
    llm = Ollama(
        model=model_name,
        temperature=0,
        base_url=''
    )
    return llm

def get_chat_gemini(model_name):
    llm = ChatGoogleGenerativeAI(
        google_api_key="",
        model=model_name,
    )
    return llm