from dataclasses import dataclass
import os
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
import google.generativeai as genai
from pydantic import BaseModel, Field


## -- Local Imports ---
# from ..Tools.finBERT import sentiment_analyzer
# from ..Tools.Mathematical_Models.Roberta import run_roberta

# Post Data Class
@dataclass
class Post:
    handle: str
    content: str

class BuffettInference:
    root_agent: SequentialAgent
    app_name: str
    user_id: str
    session_id: str

    runner: Runner
    session: InMemorySessionService

    def __init__(self):
        root_agent, APP_NAME, USER_ID, SESSION_ID = BuffettInference.initialize_pipeline()
        session_service = InMemorySessionService()

        self.user_id = USER_ID
        self.session_id = SESSION_ID
        self.runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    

    @staticmethod
    def initialize_pipeline(low = "gemini-2.0-flash", mid = "gemini-2.0-flash", high = "gemini-2.0-flash"):
        """
        Initialize the Google Generative AI API and set up the environment.
        """
        # Set up the environment variable for the API key
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAZ6XYh6LNzp32BsRV0bKJxgfEz9ziucjk"

        # Configure the Google Generative AI API
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


        # --- Constants ---
        APP_NAME = "main_pipeline_app"
        USER_ID = "Adi2p30"
        SESSION_ID = "pipeline_session_01"
        LOW_GEMINI_MODEL = low
        MID_GEMINI_MODEL = mid
        HIGH_GEMINI_MODEL = high
        # HIGH_GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"

        ## --- Initialize Tools ---
        # finBERT_tool = function_tool.FunctionTool(func=sentiment_analyzer.analyze_with_finbert)


        # Make sure to initialize ADK
        # google.adk.init(api_key=API_KEY)


        # --- Initialize Agents ---

        # Trigger Handler Agent - Using only google_search tool
        trigger_handler_agent = LlmAgent(
            name="trigger_handler_agent",
            description="You are an ADVANCED FINANCIAL THINKING AI, You are a trigger handler agent. You will receive text from a tweet or from some news source your job is to think of all the ideas and stocks surrounding this tweet explicitly mentions specific stocks affected by the trigger",
            model=HIGH_GEMINI_MODEL,
            tools=[google_search]
        )

        # Researcher Agent [SEC] - Using only google_search tool
        sec_researcher_agent = LlmAgent(
            name="sec_researcher_agent",
            description="You are a financial researcher. Researches stock behaviour and industry position and behaviour and connections to other stocks. You can use multiple sources better if you use SEC filings. But use multiple news sources to understand more information about the Stock and most affected stocks atleast 5 stocks",
            model=MID_GEMINI_MODEL,
            tools=[google_search]
        )

        # Researcher RecentNews Agent - Using only google_search tool
        recentnews_researcher_agent = LlmAgent(
            name="recentnews_researcher_agent",
            description="You are a financial researcher. Research recent financial news about the market and policy changes. Find news related to the stock and other stocks related to this stock.",
            model=MID_GEMINI_MODEL,
            tools=[google_search]
        )

        # Main Researcher Agent
        main_researcher_agent = ParallelAgent(
            name="main_researcher_agent",
            sub_agents=[sec_researcher_agent, recentnews_researcher_agent],
        )

        # Quantitative Analyst Agent - Removed nested agent tools
        quant_researcher_agent = LlmAgent(
            name="quant_researcher_agent",
            description="""You are a quantitative analyst. Your job is to analyze which quantiative prediction and machine learning models which you will execute using tools are to be used to analyze a stock and then you will interpret these results. You will then output ratings for Strong Buy or Buy or Hold or Sell or Strong Sell.
            return in this format
            {{
            "time": "<current date/time in YYYY-MM-DD HH:MM format>",
            "summary": "<Brief 1-2 line summary of the financial situation>",
            "sentiment": "<overall sentiment: Positive, Neutral, or Negative>",
            "recommendation": {{
                "<TICKER>": {{
                "sentiment": "<sentiment>",
                "rec": {{
                    "strongBuy": <percent>,
                    "buy": <percent>,
                    "hold": <percent>,
                    "sell": <percent>,
                    "strongSell": <percent>
                }},
                "reasoning": "<brief rationale for recommendation>"
                }}
            }},
            "detailed_report": "<overall explanation of the investment outlook>"
            }}
            MAKE SURE ALL STOCK TICKETS ARE ACTUAL TICKERS ON THE STOCK MARKET
            All percent values must be integers. Only return valid JSON. Do not include explanation before or after the JSON.
            """,
            model=HIGH_GEMINI_MODEL,
            tools=[google_search, ]  # Only using google_search tool, removing nested agent tools
        )

        # --- Root Agent ---
        root_agent = SequentialAgent(
            name="main_pipeline_agent",
            sub_agents=[trigger_handler_agent, main_researcher_agent, quant_researcher_agent]
        )
        return root_agent, APP_NAME, USER_ID, SESSION_ID

    def format_query(self, post: Post):
        return f"{post.handle} - {post.content}"

    def call_agent(self, post: Post):
        """
        Helper function to call the agent with a query.
        """
        query = self.format_query(post)

        content = types.Content(role='user', parts=[types.Part(text=query)])
        events = self.runner.run(
            user_id=self.user_id,
            session_id=self.session_id, 
            new_message=content)

        for event in events:
            if event.is_final_response():
                final_response = event.content.parts[0].text
                return final_response

# Example usage
if __name__ == "__main__":
    post = Post(
        "Donald Trump",
        "The United States is taking in RECORD NUMBERS in Tariffs, with the cost of almost all products going down, including gasoline, groceries, and just about everything else. Likewise, INFLATION is down. Promises Made, Promises Kept!"
    )
    
    buffett_inference = BuffettInference()
    response = buffett_inference.call_agent(post)
    print(f"Response from agent: {response}")




