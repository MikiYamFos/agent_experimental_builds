from pydantic import BaseModel
from pydantic_ai import Agent

from sql_tools import SQLTools


class SQLResult(BaseModel):
    sql_query: str
    result_text: str
    row_count: int


sql_tools = SQLTools()

agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=SQLResult,
    tools=[sql_tools.get_schema, sql_tools.run_sql],
    instructions="""
    Always call get_schema before running SQL queries.
    """,
)
