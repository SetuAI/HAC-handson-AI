from dotenv import load_dotenv
load_dotenv(override=True)
from crewai import Agent, Task ,Crew, Process
from stock_tool import get_stock_price

# agent 1 : has our custom tool

analyst = Agent(
    role = "Stock Analyst",
    goal = " Collect the latest share price and 1 month price change of a stock symbol",
    backstory = "You are a stock analyst who is interested in the latest share price.\
        You do not invent any information",
    tools = [get_stock_price], # analyst agent has access to the custom tool
    llm = "gpt-4o"
)

# agent 2 : has no tool , it only writes

writer = Agent(
    role = "Stock report writer",
    goal = "Write a short report about the share price and 1 month price change of a stock symbol",
    backstory = "You are a stock analyst who writes short reports for beginners.\
        You explain complex technical topics in simple terms with interesting analogies",
    llm = "gpt-4o"
)

# assigning task to stock analyst agent to get the share price and 1 month price change of a stock symbol
data_task = Task(
    description="Use the tool to collect the latest share price and 1 month price change for each of these stock symbols: {symbols}. \
        Call the tool once per symbol.",
    expected_output = "One line per symbol with the latest closing price and 1 month price change",
    agent = analyst,
)

# assigning task to stock report writer agent to write a short report about the share price and 1 month price change of a stock symbol
report_task = Task(
    description ="Write a short report about the share price and 1 month price change of these stock symbols: {symbols}",
    expected_output = "A markdown report with a title, 3 short paragraphs, and a conclusion",
    agent = writer,
    context = [data_task],
    output_file = "stock_report.md",
    
)

# assemble the crew
crew = Crew(
    agents = [analyst, writer],
    tasks = [data_task, report_task],
    process = Process.sequential,
    verbose = True
)

result = crew.kickoff(inputs={"symbols": "INFY.NS, TCS.NS"})
print(result)