from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# note: Gemini blows up when using TypedDict from the typing library
from pydantic import BaseModel, Field

from langchain_community.utilities import SQLDatabase

import asyncio
import os

class State(BaseModel):
    question: str = Field(..., content="question")
    query: str = Field(..., content="query")
    result: str = Field(..., content="result")
    answer: str = Field(..., content="answer")

class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: str = Field(..., content="Syntactically valid SQL query.")   

def createModel():
    # llm = HuggingFaceEndpoint(
    #     repo_id="HuggingFaceH4/zephyr-7b-beta"
    # ) 
    # return ChatHuggingFace(llm=llm, verbose=True)
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=1,
        verbose=True,
    )

llm = createModel()

# https://python.langchain.com/docs/tutorials/sql_qa/

def get_db():
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    print(f"Database dialect: {db.dialect}")
    print(db.get_usable_table_names())
    result = db.run("SELECT * FROM Artist LIMIT 10;")
    print(result)
    return db

db = get_db()

def get_query_prompt_template():
    system = """
Given an input question, create a syntactically correct {dialect} query to run to help find the answer. Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Always alias the results of aggegating functions.

Only use the following tables:
{table_info}    
    """
    
    user = """
Note: never user * in any query.

Question: {input}
    """
    
    return ChatPromptTemplate(
        [
            ("system", system),
            ("human", user),
        ]
    )

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = get_query_prompt_template().invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    print(prompt)
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result.query}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

async def sql_chain():
    question = {"question": "How many Employees are there?" }
    query = write_query(question)
    print(query["query"])
    result = execute_query(query)
    print(result["result"])
    answer = generate_answer({**question, ** query, **result})
    print(answer["answer"])

async def sql_graph():
    pass

async def main():
    print("Hello from langchain-learning!")
        
    await sql_chain()

if __name__ == "__main__":
    asyncio.run(main())
