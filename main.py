import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login

async def createChat():
    llm = HuggingFaceEndpoint(
        repo_id='microsoft/Phi-3-mini-4k-instruct',
        temperature=0.8
    ) 
    return ChatHuggingFace(llm=llm, verbose=True) 

async def chain():

    # uncomment to login to huggingface hub with access token
    # OR
    # run "huggingface-cli login"
    # login()

    chat = await createChat()

    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(
        template=template,
        input_variables=['question']
    )

    # user question
    question = "Which NFL team won the Super Bowl in the 2010 season?"

    # messages
    messages = [
    ("system", "You are a helpful assistant. Answer all questions with the knowledge you have. Be sure not to halluciate,"),
    (question)
    ]

    # ask the user question about NFL 2010
    print(chat.invoke(messages))
    
async def main():
    print("Hello from langchain-learning!")
    await chain()

if __name__ == "__main__":
    asyncio.run(main())
