# Required imports
from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Custom Tools using CrewAI's BaseTool
class WebScrapeTool(BaseTool):
    name: str = "Web Scraper"
    description: str = "Scrapes content from a given URL"

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            return f"Error scraping URL: {str(e)}"

class FileWriterTool(BaseTool):
    name: str = "File Writer"
    description: str = "Writes content to a specified file"

    def _run(self, content: str, filename: str) -> str:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote content to {filename}"
        except Exception as e:
            return f"Error writing to file: {str(e)}"

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Performs a web search using DuckDuckGo"

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

# Initialize tools
duckduckgo_search_tool = DuckDuckGoSearchTool()
web_scrape_tool = WebScrapeTool()
file_writer_tool = FileWriterTool()

# Define Agents
research_agent = Agent(
    role="Automobile Research Specialist",
    goal="Gather detailed specifications and features of automobiles",
    backstory="You're an expert in automotive technology with years of experience researching vehicle specifications.",
    verbose=True,
    llm=llm,
    tools=[duckduckgo_search_tool, web_scrape_tool]
)

pricing_agent = Agent(
    role="Automobile Pricing Analyst",
    goal="Analyze pricing trends and provide cost estimates for automobiles",
    backstory="You're a skilled pricing analyst specializing in the automotive market.",
    verbose=True,
    llm=llm,
    tools=[duckduckgo_search_tool, web_scrape_tool]
)

maintenance_agent = Agent(
    role="Maintenance Analyst",
    goal="Provide maintenance schedules and cost estimates",
    backstory="You're an experienced mechanic with expertise in vehicle maintenance.",
    verbose=True,
    llm=llm,
    tools=[duckduckgo_search_tool, web_scrape_tool]
)

performance_agent = Agent(
    role="Performance Tester",
    goal="Analyze vehicle performance metrics and capabilities",
    backstory="You're a test driver with deep knowledge of vehicle performance characteristics.",
    verbose=True,
    llm=llm,
    tools=[duckduckgo_search_tool, web_scrape_tool]
)

report_agent = Agent(
    role="Automobile Report Writer",
    goal="Create comprehensive reports based on all collected data",
    backstory="You're an experienced technical writer specializing in automobile reports.",
    verbose=True,
    llm=llm,
    tools=[file_writer_tool]
)

# Define Tasks
def create_research_task(car_model):
    return Task(
        description=f"""Research detailed specifications for the {car_model}. 
        Include:
        - Engine specifications
        - Dimensions
        - Fuel economy
        - Safety features
        - Technology features
        Use both search and web scraping tools.""",
        expected_output="A detailed summary of the car's specifications",
        agent=research_agent
    )

def create_pricing_task(car_model):
    return Task(
        description=f"""Analyze pricing for the {car_model}. 
        Include:
        - Base MSRP
        - Trim levels and prices
        - Average market price
        - Incentives/discounts
        Use web scraping for additional data.""",
        expected_output="A comprehensive pricing analysis",
        agent=pricing_agent
    )

def create_maintenance_task(car_model):
    return Task(
        description=f"""Research maintenance information for the {car_model}.
        Include:
        - Recommended maintenance schedule
        - Common issues
        - Average maintenance costs
        - Warranty information""",
        expected_output="A detailed maintenance analysis",
        agent=maintenance_agent
    )

def create_performance_task(car_model):
    return Task(
        description=f"""Analyze performance metrics for the {car_model}.
        Include:
        - Acceleration (0-60 mph)
        - Top speed
        - Handling characteristics
        - Braking distance""",
        expected_output="A comprehensive performance analysis",
        agent=performance_agent
    )

def create_report_task(car_model):
    return Task(
        description=f"""Create a comprehensive report for the {car_model} using all data.
        Include:
        - Executive summary
        - Specifications
        - Pricing analysis
        - Maintenance information
        - Performance metrics
        Save the report as '{car_model}_report.md'""",
        expected_output="A complete report in markdown format saved to file",
        agent=report_agent
    )

# Main function
def run_automobile_agent(car_model="Toyota Camry"):
    # Create tasks
    research_task = create_research_task(car_model)
    pricing_task = create_pricing_task(car_model)
    maintenance_task = create_maintenance_task(car_model)
    performance_task = create_performance_task(car_model)
    report_task = create_report_task(car_model)

    # Create crew
    automobile_crew = Crew(
        agents=[research_agent, pricing_agent, maintenance_agent, 
                performance_agent, report_agent],
        tasks=[research_task, pricing_task, maintenance_task, 
               performance_task, report_task],
        verbose=True,  # Changed from verbose=2 to verbose=True
        process="sequential"
    )

    # Execute the crew
    result = automobile_crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    try:
        result = run_automobile_agent("Tesla Model 3")
        print("\nFinal Result:")
        print(result)
        print("\nCheck the generated report file: 'Tesla Model 3_report.md'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
