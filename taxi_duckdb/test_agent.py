from sql_agent import agent
import pytest
from utils import collect_tools
from judge import assert_criteria

@pytest.mark.asyncio
async def test_agent_trip_count():
    result = await agent.run("How many trips had more than 5 passengers?")
    output = result.output

    assert isinstance(output.sql_query, str)
    assert output.sql_query.strip() != ""
    assert "22413" in output.result_text.replace(",", "")


# uv run pytest -s test_agent.py::test_agent_gets_schema_before_running_sql
@pytest.mark.asyncio
async def test_agent_gets_schema_before_running_sql():

    result = await agent.run("What is the most common payment type?")
    tool_calls = collect_tools(result.new_messages())
    tool_names = [tool.name for tool in tool_calls]

    assert tool_names[0] == "get_schema"
    assert "run_sql" in tool_names
    assert tool_names.index("get_schema") < tool_names.index("run_sql")


@pytest.mark.asyncio
async def test_agent_highest_average_fare_by_hour():

    result = await agent.run(
        "Which hour of the day has the highest average fare amount?"
    )

    await assert_criteria(
        result,
        [
            "the SQL query correctly calculates average fare by hour of day",
            "the result identifies a specific hour as having the highest average fare",
            "the result includes the actual average fare amount",
        ],
    )


def assert_tool_order(result):
    tool_calls = collect_tools(result.new_messages())
    tool_names = [tool.name for tool in tool_calls]

    assert tool_names[0] == "get_schema"
    assert "run_sql" in tool_names
    assert tool_names.index("get_schema") < tool_names.index("run_sql")


@pytest.mark.asyncio
async def test_agent_average_tip_for_credit_card_payments():
    result = await agent.run("What is the average tip amount for credit card payments?")
    assert_tool_order(result)

    await assert_criteria(
        result,
        [
            "the SQL query uses tip_amount",
            "the SQL query filters for credit card payments using payment_type",
            "the result includes the actual average tip amount",
        ],
    )


@pytest.mark.asyncio
async def test_agent_pickup_location_with_most_trips():
    result = await agent.run("Which pickup location (PULocationID) has the most trips?")
    assert_tool_order(result)

    await assert_criteria(
        result,
        [
            "the SQL query groups by PULocationID",
            "the SQL query counts trips for each pickup location",
            "the result identifies the PULocationID with the most trips",
            "the result includes the actual trip count",
        ],
    )


@pytest.mark.asyncio
async def test_agent_average_fare_for_trips_longer_than_10_miles():
    result = await agent.run("What is the average fare for trips longer than 10 miles?")
    assert_tool_order(result)

    await assert_criteria(
        result,
        [
            "the SQL query uses fare_amount",
            "the SQL query filters trips where trip_distance is greater than 10",
            "the result includes the actual average fare amount",
        ],
    )


@pytest.mark.asyncio
async def test_agent_zero_passenger_trips():
    result = await agent.run("How many trips had zero passengers recorded?")
    assert_tool_order(result)

    await assert_criteria(
        result,
        [
            "the SQL query filters on passenger_count",
            "the SQL query counts trips where passenger_count equals 0",
            "the result includes the actual number of trips with zero passengers recorded",
        ],
    )


@pytest.mark.asyncio
async def test_agent_busiest_day_of_week():
    result = await agent.run("What is the busiest day of the week for taxi trips?")
    assert_tool_order(result)

    await assert_criteria(
        result,
        [
            "the SQL query extracts or calculates day of week from tpep_pickup_datetime",
            "the SQL query counts trips by day of week",
            "the result identifies the busiest day of the week",
            "the result includes the actual trip count",
        ],
    )
