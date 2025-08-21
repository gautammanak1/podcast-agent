#!/usr/bin/env python3
"""
Automated Podcast Generation System using LangGraph

This script converts the Jupyter notebook into a Python agent that takes queries as input
and generates podcast content based on the given topic.

Usage:
1. Set environment variables or create .env file:
   export GEMINI_API_KEY='your_key'
   export OPENAI_API_KEY='your_key'
   export TAVILY_API_KEY='your_key'
   
2. Run: python podcast_agent.py
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from uagents_adapter import LangchainRegisterTool, cleanup_uagent
import operator
from datetime import datetime
from typing import Any, Annotated, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import tiktoken

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string

from langgraph.constants import Send
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever


class PodcastAgent:
    """Main Podcast Generation Agent"""
    
    def __init__(self):
        """Initialize the agent with API keys from environment variables"""
        self.setup_api_keys()
        self.setup_models()
        self.setup_token_management()
        self.build_graphs()
    
    def setup_api_keys(self):
        """Configure API keys from environment variables"""
        # Get API keys from environment variables
        openai_key = os.getenv('OPENAI_API_KEY')
        tavily_key = os.getenv('TAVILY_API_KEY')
        langchain_key = os.getenv('LANGCHAIN_API_KEY', '')
        
        # Validate required keys
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        
        print(f"✓ Using OpenAI model: gpt-3.5-turbo")
        print(f"✓ LangSmith tracing: {'enabled' if langchain_key else 'disabled'}")
        
        # Only enable LangSmith tracing if API key is provided
        if langchain_key:
            os.environ["LANGCHAIN_API_KEY"] = langchain_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "PodcastGenAI"
        else:
            # Disable LangSmith tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    def get_model(self, model: str = "gpt-3.5-turbo", temp: float = 0.1, max_tokens: int = 100):
        """Get model from OpenAI"""
        model = ChatOpenAI(
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
        )
        return model
    
    def setup_models(self):
        """Setup required models"""
        # Use OpenAI for all podcast generation
        self.podcast_model = self.get_model("gpt-3.5-turbo", temp=0.21, max_tokens=2000)
    
    def setup_token_management(self):
        """Setup token counting and management"""
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = 14000  # Leave buffer for response
        self.chunk_overlap = 200  # Overlap between chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = None) -> List[str]:
        """Split text into chunks that fit within token limits"""
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # If text fits within limit, return as is
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        # Split by sections first (double newlines)
        sections = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # Check if adding this section would exceed limit
            test_chunk = current_chunk + "\n\n" + section if current_chunk else section
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single section is too large, split by sentences
                if self.count_tokens(section) > max_tokens:
                    sentences = section.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        test_sentence = temp_chunk + ". " + sentence if temp_chunk else sentence
                        
                        if self.count_tokens(test_sentence) <= max_tokens:
                            temp_chunk = test_sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def summarize_text_chunks(self, chunks: List[str], system_prompt: str) -> str:
        """Process multiple chunks and combine results"""
        if len(chunks) == 1:
            # Single chunk, process normally
            try:
                messages = [SystemMessage(content=system_prompt + chunks[0])]
                response = self.podcast_model.invoke(messages)
                return response.content
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    # Even single chunk is too large, use GPT for summarization
                    return self._fallback_summarize(chunks[0], system_prompt)
                raise e
        
        # Multiple chunks - summarize each and combine
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_prompt = f"{system_prompt}\n\nThis is part {i+1} of {len(chunks)} parts. Focus on the key points:\n\n{chunk}"
                messages = [SystemMessage(content=chunk_prompt)]
                response = self.podcast_model.invoke(messages)
                summaries.append(response.content)
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    summary = self._fallback_summarize(chunk, system_prompt)
                    summaries.append(summary)
                else:
                    raise e
        
        # Combine summaries
        combined_summary = "\n\n".join(summaries)
        
        # If combined summary is still too long, summarize again
        if self.count_tokens(combined_summary) > self.max_tokens:
            final_prompt = f"{system_prompt}\n\nCombine and synthesize these summaries into a cohesive final version:\n\n{combined_summary}"
            messages = [SystemMessage(content=final_prompt)]
            response = self.podcast_model.invoke(messages)
            return response.content
        
        return combined_summary
    
    def _fallback_summarize(self, text: str, system_prompt: str) -> str:
        """Fallback using OpenAI for very large chunks"""
        try:
            gpt_model = self.get_model("gpt-3.5-turbo", 0.3, 1000)
            messages = [SystemMessage(content=f"{system_prompt}\n\nSummarize the key points from this content:\n\n{text[:8000]}...")]
            response = gpt_model.invoke(messages)
            return response.content
        except Exception as e:
            # Last resort - truncate
            return f"Content too large to process fully. Key excerpt: {text[:2000]}..."
    
    def build_graphs(self):
        """Build the planning and main graphs"""
        self.build_planning_graph()
        self.build_interview_graph()
        self.build_main_graph()
    
    def build_planning_graph(self):
        """Build the planning subgraph"""
        
        class Planning(TypedDict):
            topic: str
            keywords: list[str]
            subtopics: list[str]
        
        class Keywords(BaseModel):
            """Answer with at least 5 keywords that you think are related to the topic"""
            keys: list = Field(description="list of at least 5 keywords related to the topic")
        
        class Subtopics(BaseModel):
            """Answer with at least 5 subtopics related to the topic"""
            subtopics: list = Field(description="list of at least 5 subtopics related to the topic")
        
        class Structure(BaseModel):
            """Structure of the podcast having in account the topic and the keywords"""
            subtopics: list[Subtopics] = Field(description="5 subtopics that we will review in the podcast related to the Topic and the Keywords")
        
        gpt_keywords = self.get_model("gpt-3.5-turbo", 0.1, 50)
        model_keywords = gpt_keywords.with_structured_output(Keywords)
        
        gpt_structure = self.get_model("gpt-3.5-turbo", 0.3, 1000)
        model_structure = gpt_structure.with_structured_output(Structure)
        
        def get_keywords(state: Planning):
            topic = state['topic']
            messages = [SystemMessage(content="You task is to generate 5 relevant words about the following topic: " + topic)]
            message = model_keywords.invoke(messages)
            return {'keywords': message.keys}
        
        def get_structure(state: Planning):
            topic = state['topic']
            keywords = state['keywords']
            messages = [SystemMessage(content="You task is to generate 5 subtopics to make a podcast about the following topic: " + topic + "and the following keywords:" + " ".join(keywords))]
            message = model_structure.invoke(messages)
            return {"subtopics": message.subtopics[0].subtopics}
        
        plan_builder = StateGraph(Planning)
        plan_builder.add_node("Keywords", get_keywords)
        plan_builder.add_node("Structure", get_structure)
        plan_builder.add_edge(START, "Keywords")
        plan_builder.add_edge("Keywords", "Structure")
        plan_builder.add_edge("Structure", END)
        
        self.graph_plan = plan_builder.compile()
    
    def build_interview_graph(self):
        """Build the interview subgraph"""
        
        class InterviewState(MessagesState):
            topic: str
            max_num_turns: int
            context: Annotated[list, operator.add]
            section: str
            sections: list
        
        class SearchQuery(BaseModel):
            search_query: str = Field(None, description="Search query for retrieval.")
        
        podcast_gpt = self.get_model(max_tokens=1000)
        structured_llm = podcast_gpt.with_structured_output(SearchQuery)
        
        question_instructions = """You are the host of a popular podcast and you are tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {topic}
        #
Begin by introducing the topic that fits your goals, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help"

Remember to stay in character throughout your response"""
        
        def generate_question(state: InterviewState):
            """Node to generate a question"""
            topic = state["topic"]
            messages = state["messages"]
            
            system_message = question_instructions.format(topic=topic)
            question = podcast_gpt.invoke([SystemMessage(content=system_message)] + messages)
            
            return {"messages": [question]}
        
        search_instructions = SystemMessage(content=f"""You will be given a conversation between a host of a popular podcast and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query""")
        
        def search_web(state: InterviewState):
            """Retrieve docs from web search"""
            search_query = structured_llm.invoke([search_instructions] + state['messages'])
            
            tavily_search = TavilySearchResults(max_results=3)
            search_docs = tavily_search.invoke(search_query.search_query)
            
            formatted_search_docs = "\n\n---\n\n".join([
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ])
            
            return {"context": [formatted_search_docs]}
        
        def search_wikipedia(state: InterviewState):
            """Retrieve docs from web search (Wikipedia alternative)"""
            search_query = structured_llm.invoke([search_instructions] + state['messages'])
            
            # Use Tavily search with Wikipedia focus instead
            tavily_search = TavilySearchResults(max_results=2)
            wiki_query = f"site:en.wikipedia.org {search_query.search_query}"
            search_docs = tavily_search.invoke(wiki_query)
            
            formatted_search_docs = "\n\n---\n\n".join([
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ])
            
            return {"context": [formatted_search_docs]}
        
        answer_instructions = """You are an expert being interviewed by a popular podcast host.
Here is the analyst's focus area: {topic}.
Your goal is to answer a question posed by the interviewer.
To answer the question, use this context:
{context}
When answering questions, follow these steps

1. Use only the information provided in the context.

2. Do not introduce outside information or make assumptions beyond what is explicitly stated in the context.

3. The context includes sources on the topic of each document.

4. Make it interesting."""
        
        def generate_answer(state: InterviewState):
            """Node to answer a question"""
            topic = state["topic"]
            messages = state["messages"]
            context = state["context"]
            
            system_message = answer_instructions.format(topic=topic, context=context)
            answer = podcast_gpt.invoke([SystemMessage(content=system_message)] + messages)
            
            answer.name = "expert"
            
            return {"messages": [answer]}
        
        def save_podcast(state: InterviewState):
            """save_podcast"""
            messages = state["messages"]
            interview = get_buffer_string(messages)
            return {"section": interview}
        
        def route_messages(state: InterviewState, name: str = "expert"):
            """Route between question and answer"""
            messages = state["messages"]
            max_num_turns = state.get('max_num_turns', 2)
            
            num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
            
            if num_responses >= max_num_turns:
                return 'Save podcast'
            
            last_question = messages[-2]
            
            if "Thank you so much for your help" in last_question.content:
                return 'Save podcast'
            return "Host question"
        
        section_writer_instructions = """You are an expert technical writer.

Your task is to create an interesting, easily digestible section of a podcast based on an interview.

1. Analyze the content of the interview

2. Create a script structure using markdown formatting

3. Make your title engaging based upon the focus area of the analyst:
{focus}

4. For the conversation:
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Mention turns of "Interviewer" and "Expert"
- Aim for approximately 1000 words maximum

5. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""
        
        def write_section(state: InterviewState):
            """Node to answer a question"""
            section = state["section"]
            topic = state["topic"]
            
            system_message = section_writer_instructions.format(focus=topic)
            full_prompt = system_message + f"Use this source to write your section: {section}"
            messages = [SystemMessage(content=full_prompt)]
            section_res = self.podcast_model.invoke(messages)
            
            return {"sections": [section_res.content]}
        
        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("Host question", generate_question)
        interview_builder.add_node("Web research", search_web)
        interview_builder.add_node("Wiki research", search_wikipedia)
        interview_builder.add_node("Expert answer", generate_answer)
        interview_builder.add_node("Save podcast", save_podcast)
        interview_builder.add_node("Write script", write_section)
        
        interview_builder.add_edge(START, "Host question")
        interview_builder.add_edge("Host question", "Web research")
        interview_builder.add_edge("Host question", "Wiki research")
        interview_builder.add_edge("Web research", "Expert answer")
        interview_builder.add_edge("Wiki research", "Expert answer")
        interview_builder.add_conditional_edges("Expert answer", route_messages, ['Host question', 'Save podcast'])
        interview_builder.add_edge("Save podcast", "Write script")
        interview_builder.add_edge("Write script", END)
        
        memory = MemorySaver()
        self.podcast_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Create podcast")
    
    def build_main_graph(self):
        """Build the main research graph"""
        
        class ResearchGraphState(TypedDict):
            topic: Annotated[str, operator.add]
            keywords: List
            max_analysts: int
            subtopics: List
            sections: Annotated[list, operator.add]
            introduction: str
            content: str
            conclusion: str
            final_report: str
        
        report_writer_instructions = """You are a podcast script writer preparing a script for an episode on this overall topic:

{topic}

You have a dedicated researcher who has delved deep into various subtopics related to the main theme.
Your task:

1. You will be given a collection of part of script podcast from the researcher, each covering a different subtopic.
2. Carefully analyze the insights from each script.
3. Consolidate these into a crisp and engaging narrative that ties together the central ideas from all of the script, suitable for a podcast audience.
4. Weave the central points of each script into a cohesive and compelling story, ensuring a natural flow and smooth transitions between segments, creating a unified and insightful exploration of the overall topic.

To format your script:

1. Use markdown formatting.
2. Write in a conversational and engaging tone suitable for a podcast.
3. Seamlessly integrate the insights from each script into the narrative, using clear and concise language.
4. Use transitional phrases and signposting to guide the listener through the different subtopics.

Here are the scripts from the researcher to build your podcast script from:

{context}"""
        
        intro_instructions = """You are a podcast producer crafting a captivating introduction for an upcoming episode on {topic}.
You will be given an outline of the episode's main segments.
Your job is to write a compelling and engaging introduction that hooks the listener and sets the stage for the discussion.
Include no unnecessary preamble or fluff.
Target around 300 words, using vivid language and intriguing questions to pique the listener's curiosity and preview the key themes and topics covered in the episode.
Use markdown formatting.
Create a catchy and relevant title for the episode and use the # header for the title.
Use ## Introduction as the section header for your introduction.
Here are the segments to draw upon for crafting your introduction: {formatted_str_sections}"""
        
        conclusion_instructions = """You are a podcast producer crafting a memorable conclusion for an episode on {topic}.
You will be given an outline of the episode's main segments.
Your job is to write a concise and impactful conclusion that summarizes the key takeaways and leaves a lasting impression on the listener.
Include no unnecessary preamble or fluff.
Target around 200 words, highlighting the most important insights and offering a thought-provoking closing statement that encourages further reflection or action.
Use markdown formatting.
Use ## Conclusion as the section header for your conclusion.
Here are the segments to draw upon for crafting your conclusion: {formatted_str_sections}"""
        
        def initiate_all_interviews(state: ResearchGraphState):
            """This is the "map" step where we run each interview sub-graph using Send API"""
            topic = state["topic"]
            return [Send("Create podcast", {"topic": topic,
                                          "messages": [HumanMessage(
                                              content=f"So you said you were researching about {subtopic}?"
                                          )]}) for subtopic in state["subtopics"]]
        
        def write_report(state: ResearchGraphState):
            sections = state["sections"]
            topic = state["topic"]
            
            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
            system_message = report_writer_instructions.format(topic=topic, context="")
            
            # Check token count and chunk if necessary
            full_content = system_message + formatted_str_sections
            if self.count_tokens(full_content) > self.max_tokens:
                # Chunk the sections and process
                chunks = self.chunk_text(formatted_str_sections)
                report_text = self.summarize_text_chunks(chunks, system_message)
            else:
                # Process normally
                try:
                    final_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
                    messages = [SystemMessage(content=final_message)]
                    report = self.podcast_model.invoke(messages)
                    report_text = report.content
                except Exception as e:
                    if "context_length_exceeded" in str(e):
                        chunks = self.chunk_text(formatted_str_sections)
                        report_text = self.summarize_text_chunks(chunks, system_message)
                    else:
                        raise e
            
            return {"content": report_text}
        
        def write_introduction(st ate: ResearchGraphState):
            sections = state["sections"]
            topic = state["topic"]
            
            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
            base_instructions = intro_instructions.format(topic=topic, formatted_str_sections="")
            
            # Check token count and chunk if necessary
            full_content = base_instructions + formatted_str_sections
            if self.count_tokens(full_content) > self.max_tokens:
                # Chunk the sections and process
                chunks = self.chunk_text(formatted_str_sections)
                intro_text = self.summarize_text_chunks(chunks, base_instructions)
            else:
                # Process normally
                try:
                    instructions = intro_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
                    messages = [SystemMessage(content=instructions)]
                    intro = self.podcast_model.invoke(messages)
                    intro_text = intro.content
                except Exception as e:
                    if "context_length_exceeded" in str(e):
                        chunks = self.chunk_text(formatted_str_sections)
                        intro_text = self.summarize_text_chunks(chunks, base_instructions)
                    else:
                        raise e
            
            return {"introduction": intro_text}
        
        def write_conclusion(state: ResearchGraphState):
            sections = state["sections"]
            topic = state["topic"]
            
            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
            base_instructions = conclusion_instructions.format(topic=topic, formatted_str_sections="")
            
            # Check token count and chunk if necessary
            full_content = base_instructions + formatted_str_sections
            if self.count_tokens(full_content) > self.max_tokens:
                # Chunk the sections and process
                chunks = self.chunk_text(formatted_str_sections)
                conclusion_text = self.summarize_text_chunks(chunks, base_instructions)
            else:
                # Process normally
                try:
                    instructions = conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
                    messages = [SystemMessage(content=instructions)]
                    conclusion = self.podcast_model.invoke(messages)
                    conclusion_text = conclusion.content
                except Exception as e:
                    if "context_length_exceeded" in str(e):
                        chunks = self.chunk_text(formatted_str_sections)
                        conclusion_text = self.summarize_text_chunks(chunks, base_instructions)
                    else:
                        raise e
            
            return {"conclusion": conclusion_text}
        
        def finalize_report(state: ResearchGraphState):
            """The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""
            content = state["content"]
            final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
            
            return {"final_report": final_report}
        
        def start_parallel(state):
            """No-op node that should be interrupted on"""
            pass
        
        builder = StateGraph(ResearchGraphState)
        builder.add_node("Planing", self.graph_plan)
        builder.add_node("Start research", start_parallel)
        builder.add_node("Create podcast", self.podcast_graph)
        builder.add_node("Write report", write_report)
        builder.add_node("Write introduction", write_introduction)
        builder.add_node("Write conclusion", write_conclusion)
        builder.add_node("Finalize podcast", finalize_report)
        
        builder.add_edge(START, "Planing")
        builder.add_edge("Planing", "Start research")
        builder.add_conditional_edges("Start research", initiate_all_interviews, ["Planing", "Create podcast"])
        builder.add_edge("Create podcast", "Write report")
        builder.add_edge("Create podcast", "Write introduction")
        builder.add_edge("Create podcast", "Write conclusion")
        builder.add_edge(["Write introduction", "Write report", "Write conclusion"], "Finalize podcast")
        builder.add_edge("Finalize podcast", END)
        
        memory = MemorySaver()
        self.main_graph = builder.compile(checkpointer=memory)
    
    def generate_podcast(self, topic: str, thread_id: str = "1") -> str:
        """Generate a podcast based on the given topic"""
        input_g = {"topic": topic}
        thread = {"configurable": {"thread_id": thread_id}}
        
        print(f"Generating podcast for topic: {topic}")
        print("Processing...")
        
        for event in self.main_graph.stream(input_g, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            print(f"Processing: {node_name}")
        
        final_state = self.main_graph.get_state(thread)
        report = final_state.values.get('final_report')
        
        return report


# Initialize the global agent instance
agent = None

def podcast_agent_func(query):
    """Wrapper function for UAgent integration"""
    global agent
    
    # Handle input if it's a dict with 'input' key
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    try:
        # Initialize agent if not already done
        if agent is None:
            agent = PodcastAgent()
        
        # Generate podcast and return markdown directly
        result = agent.generate_podcast(query)
        return result if result else "Failed to generate podcast content"
        
    except Exception as e:
        error_msg = str(e)
        if "context_length_exceeded" in error_msg:
            return f"Error: Content too large for processing. The research generated too much content to fit in the model's context window. Try a more specific topic or reduce the scope of research."
        elif "rate_limit" in error_msg.lower():
            return f"Error: API rate limit reached. Please try again in a few moments."
        elif "invalid_api_key" in error_msg.lower():
            return f"Error: Invalid API key. Please check your OpenAI API configuration."
        else:
            return f"Error generating podcast: {error_msg}"

def main():
    """Main function for UAgent registration"""
    
    # Get API token for Agentverse
    API_TOKEN = os.environ.get("AGENTVERSE_API_KEY")
    
    if not API_TOKEN:
        raise ValueError("Please set AGENTVERSE_API_KEY environment variable")
    
    try:
        # Test agent initialization
        test_agent = PodcastAgent()
        print("✓ Agent initialized successfully")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set the required environment variables:")
        print("export OPENAI_API_KEY='your_openai_key'")
        print("export TAVILY_API_KEY='your_tavily_key'")
        print("export AGENTVERSE_API_KEY='your_agentverse_key'")
        print("export LANGCHAIN_API_KEY='your_langchain_key'  # Optional")
        return
    
    # Register the podcast agent via uAgent
    tool = LangchainRegisterTool()
    agent_info = tool.invoke(
        {
            "agent_obj": podcast_agent_func,
            "name": "PodScribe Agent",
            "port": 8080,
            "description": "An AI agent that generates podcast scripts on any topic with research and structured content",
            "api_token": API_TOKEN,
            "mailbox": True,
            "ai_agent_address":"agent1qdafg4whsrjrpzgmn47pvus9g7uwzn38jrql4cese7dg54nkuqwtj2kkztt"
        }
    )
    
    print(f"✅ Registered Podcast Generation Agent: {agent_info}")
    
    # Keep the agent alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down Podcast Generation Agent...")
        cleanup_uagent("podcast_generation_agent")
        print("✅ Agent stopped.")


if __name__ == "__main__":
    main()
