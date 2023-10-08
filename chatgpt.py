import os
import sys

import constants
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = constants.API_KEY

query = sys.argv[1]

# Add folder or change "../data" to suit your needs
loader = DirectoryLoader("../data", glob="*.txt")
print(loader.load())
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI()))