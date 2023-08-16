import os
from langchain import OpenAI, LLMChain, ConversationChain
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory


def init():
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') is None or os.getenv('HUGGINGFACEHUB_API_TOKEN') is None:
        print("OPEN API OR HUGGING FACE API KEYS NOT SET")
    else:
        print("KEYS ARE SET!")


def openai_example():
    llm = OpenAI(temperature=0.9)
    text = "Who is warren buffet?"
    # print(llm.predict(text))
    print(llm(text))


def huggingface_example():
    llm = HuggingFaceHub(repo_id="google/flan-t5-large")
    print(llm("translate english to spanish for - how are you?"))


def prompt_template():
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}")
    #print(prompt.format(product="microwave"))
    llm = OpenAI(temperature=0.9)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = chain.run("aeroplane")
    print(response)

def prompt_buffer_memory():
    llm = OpenAI(temperature=0.9)
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, verbose=True, memory=memory)
    response1 = chain.run("Name me some famous french food?")
    print(response1)
    response2 = chain.run("Can you name some of their pastries")
    print(response2)
    response3 = chain.run("where are these people from?")
    print(response3)
    print(memory)

def prompt_buffer_memory_window():
    prompt = PromptTemplate.from_template("What are the menu items for {food} restaurant?")
    llm = OpenAI(temperature=0.9)
    memory = ConversationBufferWindowMemory(k=1)
    chain = ConversationChain(llm=llm, verbose=True, memory=memory)
    response1 = chain.run("Name me some famous french food?")
    print(response1)
    response2 = chain.run("Can you name some of their pastries")
    print(response2)
    response3 = chain.run("where are these people from?")
    print(response3)
    print(memory)


def sequential_chain_prompt_template():
    llm = OpenAI(temperature = 0.8)
    genre_prompt = PromptTemplate.from_template("I want to direct a {genre} movie. Suggest me a name for this movie?")
    genre_chain = LLMChain(llm = llm, prompt = genre_prompt, verbose =True, output_key="movie_name")
    movie_names_prompt = PromptTemplate.from_template("Name some movies of {genre} genre" )
    movie_name_chain = LLMChain(llm=llm, prompt=movie_names_prompt, verbose = True, output_key="movie_collection")
    chain =SequentialChain(
        chains=[genre_chain, movie_name_chain],
        input_variables = ['genre'],
        output_variables = ['movie_name', 'movie_collection']
    )
    print(chain({'genre':'Thriller'}))
    print(chain.memory)

def agent_tool():
    from langchain.agents import AgentType, initialize_agent, load_tools
    from langchain.llms import OpenAI
    llm = OpenAI(temperature=0.7)
    tools = load_tools(['serpapi','llm-math'], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("Who won the wimbledon in 2023?")


def main():
    init()
    ## Uncomment each to test simple functions
    #openai_example()
    #huggingface_example()
    #prompt_template()
    #sequential_chain_prompt_template()
    #prompt_buffer_memory()
    #prompt_buffer_memory_window()
    agent_tool()


if __name__ == '__main__':
    main()
