from flask import Flask, request, jsonify, session
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import os
import torch
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from transformers.pipelines import  pipeline
import warnings
warnings.filterwarnings('ignore')

# Instantiate Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dheygpt_boat'

# Define the template for the prompt
prompt_template_twitter = """
Generate a Twitter post sequence based on the following details:

Current conversation context: {history}
Most recent input: {input}

Requirements:
- Write the tweet in English.
- Ensure the content is engaging and concise, suitable for Twitter’s character limit.
- Use emojis to enhance the expression where appropriate.
- Include relevant hashtags to increase visibility and engagement.

Note: Keep the tone light and fun, and make sure the tweet is relevant to the current topic.
"""
prompt_template_linkedin = """
Generate a professional LinkedIn post based on the following details:

Current conversation context: {history}

Post topic provided by user:
Human: {input}

Requirements:
- The language should be professional yet inviting to attract industry professionals, researchers, and enthusiasts.
- Convey excitement about the event or topic to spark interest and engagement.
- Highlight key topics that are relevant and thought-provoking to encourage discussions among the LinkedIn community.
- Use emojis sparingly to add a touch of enthusiasm without compromising the professional tone.
- Include relevant hashtags to enhance visibility and networking potential if appropriate.

Note: Ensure the post is succinct and impactful, designed to stimulate interaction and professional dialogue.
"""
prompt_template_proposal = """
Generate a detailed proposal based on the following specifications:

Current understanding of the situation or problem: {history}

Specific request or need outlined by the user:
Human: {input}

Requirements:
- The proposal should be clear and well-structured, detailing the objectives, methods, expected outcomes, and benefits of the proposed idea or project.
- Use formal and persuasive language to convincingly present the proposal to stakeholders or decision-makers.
- Include key data points or evidence that support the feasibility and potential impact of the proposal.
- Emphasize how the proposal aligns with the goals or interests of the audience, highlighting any unique advantages or opportunities it presents.

Note: Ensure that the proposal is comprehensive yet concise, making it easy for the reader to understand the key messages and make informed decisions.
"""
prompt_template_question_answering = """
Generate a response to the following query based on the provided context and the specific question asked by the user:

Current understanding of the topic: {history}

User's query:
Human: {input}

Requirements:
- Provide a clear and concise answer that directly addresses the user's question.
- Include relevant facts or explanations to ensure the response is informative.
- Use a neutral and professional tone suitable for a broad audience.
- If the question involves complex topics or concepts, simplify the explanation without losing essential details.
- Whenever applicable, offer additional resources or directions for further exploration of the topic.

Note: Aim to educate and clarify, ensuring the user gains a deeper understanding of the questions.
"""


# MODEL_NAME = os.path.join("models",f"{model_name}")




# Define the Flask route
@app.route('/conversation', methods=['POST'])
def converse():
    model_name = request.json.get('model_name')
    user_input = request.json.get('prompt')
    prompt_type = request.json.get('contentfor')  
    history = session.get('history', [])
    history.append(user_input)
    session['history'] = history
    session['input'] = user_input

    model_path = os.path.join(f"/app/models/{model_name}")

    quantization_config = BitsAndBytesConfig(

    load_in_16bit=True,
    bnb_16bit_compute_dtype=torch.float16,
    bnb_16bit_quant_type="nf4",
    bnb_16bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
        quantization_config=quantization_config
    )

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.max_new_tokens = 4096
    #has made change here
    generation_config.temperature = 0.4
    generation_config.top_p = 0.5
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
    )
    llm = HuggingFacePipeline(
    pipeline=text_gen_pipeline,
)
    #session['model_name'] = model_name
    # Select the appropriate prompt based on user input
    if prompt_type == 'linkedin': 
        prompt = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_linkedin)
    elif prompt_type=='twitter':
        prompt = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_twitter)
    elif prompt_type=='framing': #for Proposal Generation
        prompt = PromptTemplate(input_variables=['history', 'input'], template=prompt_template_proposal)
    elif prompt_type=='qna':
        prompt = PromptTemplate(input_variables=['history','input'],template=prompt_template_question_answering)
    # Instantiate the ConversationChain with the selected prompt
    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        verbose=False,
        memory=ConversationBufferWindowMemory(k=2)
    )
    
    llm_response = conversation.run(history=history, input=user_input)
    return jsonify({'response': llm_response})

# Run the Flask app
if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5002, debug=True)
