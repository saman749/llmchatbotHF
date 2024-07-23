"""
File for making the chatbot directly on HuggingFace spaces.

"""
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from transformers import pipeline

# %% method 1: Using langchain.
# This method needs you downloading the language model locally.
# and installing `langchain` library.

## Function To get response from LLAma 2 model
def get_llama_response_langchain(
        input_text: str,
        no_words: str,
        blog_style: str,
        config: dict,
    ):
    ### LLama2 model
    llm = CTransformers(model='marella/gpt-2-ggml', config=config)
    ## Prompt Template
    template="""
        Explain to {blog_style}, about {input_text}
        within {no_words} words.
            """
    prompt=PromptTemplate(
        input_variables=["blog_style","input_text",'no_words'],
        template=template,
    )
    
    ## Generate the ressponse from the LLama 2 model
    response = llm(
        prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            no_words=no_words,
        ),
    )
    print(response)
    return response


# %% Using HuggingFace.
# Function to get response from LLama 2 model using HuggingFace
def get_llama_response_huggingface(input_text: str, no_words: str, blog_style: str, config: dict):
    """
    Get LLama's response using HuggingFace.
    """
    # Load the model and tokenizer from HuggingFace
    model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    generator = pipeline('text-generation', model=model_name, token=HF_TOKEN)
    print('here')
    # Create the prompt
    prompt = f"Write a blog for {blog_style} audience for a topic {input_text} within {no_words} words."
    print('now here')
    # Generate the response
    response = generator(
        prompt,
        max_length=int(no_words),
        num_return_sequences=1,
    )
    print('and here')
    # Extract the generated text
    generated_text = response[0]['generated_text']
    return generated_text

# %% Using what the YouTube video is doing exactly.
def get_llama_response_youtube(
      input_text: str,
      no_words: str,
      audience: str,
      config: dict,
    ) -> st:

    llm = CTransformers(
        model = 'TheBloke/Llama-2-7B-Chat-GGML',
        model_type='llama',
        config=config,
    )

    ## Prompt Template
    template="""
        Explain to {audience} about {input_text}, within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["audience", "input_text", 'no_words'],
        template=template,
    )

    ## Generate the ressponse
    response = llm(prompt.format(
        audience=audience,
        input_text=input_text,
        no_words=no_words,
    ))
    print(response)
    return response


# %% main code.

## Page Configuration
st.set_page_config(page_title="Explain PID control to students",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Explain PID controllers")
input_text=st.text_input(
    "Enter the title that you want to learn about in PID control:",
)

# Creating two more columns for additional 2 fields.
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox(
        'Writing the explanation for',
        ('electrical engineering students',
         'mechanical engineering students',
         'chemical engineering students',
        ),
        index=0,
    )
submit = st.button("Generate")


## user selections.
config = {'max_new_tokens':256, 'temperature':0.01}
llm_method = 'exact'  # or 'langchain' or 'huggingface'

if llm_method=='huggingface':
    llama_response = get_llama_response_huggingface(input_text, no_words, blog_style, config)
elif llm_method=='langchain':
    llama_response = get_llama_response_langchain(input_text, no_words, blog_style, config)
elif llm_method=='exact':
    llama_response = get_llama_response_youtube(input_text, no_words, blog_style, config)
else:
    raise ValueError('wrong llm method!')

# Final response
if submit:
    st.write(llama_response)
