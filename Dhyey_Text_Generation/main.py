import torch
#import gradio as gr
from textwrap import fill


from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    )

from langchain import PromptTemplate
from langchain import HuggingFacePipeline

from langchain.vectorstores import Chroma
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredURLLoader
# has made change here
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from transformers.pipelines import  pipeline
import warnings
warnings.filterwarnings('ignore')

# from ctransformers import AutoModelForCausalLM
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from torch import cuda

# Define the template for the prompt
# Callbacks support token-wise streaming


prompt_template_twitter = """Generate a list of two varied versions of Twittter post sequences. \
    The topic of the post is as follows:
    
    current conversation:{history} 
    Human: {input}
    

     You are twitter expert,You are required to write tweatd  in English. Keep it fun to read by adding some emojis and supporting hashtags (just if you think it's necessary).

    
    """
    

prompt_template_linkedin=""" Generate a professional Linkedin post 
    current conversation:{history}
    
    The topic of the post is as follows:
    Human:{input}
    
    The post should aims to attract industry professionals, researchers, and enthusiasts
    The post should convey the excitement surrounding the event
    highlight key topics and encourage engagement from the LinkedIn community (just if you think it's necessary), 
    Keep it fun to read by adding some emojis and supporting hashtags (just if you think it's necessary).
    
    """

# Instantiate the PromptTemplate with the template and input variables
prompt_linkedin = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_linkedin)
prompt_twitter=PromptTemplate(input_variables=['history','input'],template=prompt_template_twitter)
#!pip install bitsandbytes

MODEL_NAME =r"D:\Dhyey text generation\Mistral-7B-Instruct-v0.2"

quantization_config = BitsAndBytesConfig(
    load_in_16bit=True,
    bnb_16bit_compute_dtype=torch.float16,
    bnb_16bit_quant_type="nf4",
    bnb_16bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cuda",
    quantization_config=quantization_config
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 4096
#has made change here
generation_config.temperature = 0.5
generation_config.top_p = 0.5
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
)
llm = HuggingFacePipeline(
    pipeline=pipeline,

    
    )

conversation = ConversationChain(
    prompt=prompt_linkedin,
    llm=llm,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2)
)

# Define a function to interactively run the conversation
def run_conversation():
    history = []
    while True:
        user_input = input("You: ")
        history.append(user_input)
        llm_response = conversation.run(history=history, input=user_input)
        print("AI:", llm_response)
        history.append(llm_response)

# Run the conversation
if __name__ == "__main__":
    run_conversation()
