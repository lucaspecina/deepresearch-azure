"""
ReAct agent implementation for DeepResearch.
Uses a reasoning-action-observation cycle to solve tasks.
"""

import re
import json
import logging
from openai import AzureOpenAI
import deepresearch_azure.config as config
from deepresearch_azure.search_tools import get_all_tools
from deepresearch_azure.prompts import REACT_PROMPT

class ReActAgent:
    """
    ReAct agent that uses a reasoning-action-observation cycle.
    """
    
    def __init__(self, verbose=False):
        """Initialize the ReAct agent"""
        self.tools = {tool.name: tool for tool in get_all_tools()}
        
        # Setup logging
        self.verbose = verbose
        self.logger = logging.getLogger('deepresearch.agent')
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # Initialize OpenAI client
        self.client = AzureOpenAI(
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_ENDPOINT
        )
        
        self.model = config.AGENT_MODEL_DEPLOYMENT
        
        # Format the tools for the prompt
        self.tools_description = self._format_tools_for_prompt()
        
        # Track which tools have been used
        self.used_tools = set()
        
        # Initialize conversation context
        self.context = []
        self.original_query = None
        
        # Set min iterations before final answer (to encourage tool use)
        self.min_iterations = 2
        
        self.logger.info(f"ReAct agent initialized with model: {self.model}")
        print(f"ReAct agent initialized with model: {self.model}")
        self.logger.info(f"Available tools: {', '.join(self.tools.keys())}")
        print(f"Available tools: {', '.join(self.tools.keys())}")
        
    def _format_tools_for_prompt(self):
        """Format the available tools for the prompt"""
        tools_list = []
        for name, tool in self.tools.items():
            tool_str = f"- {name}: {tool.description}\n"
            tool_str += f"  Takes inputs: {{'query': 'The search query to execute'}}\n"
            tool_str += f"  Returns an output of type: string"
            tools_list.append(tool_str)
            
        # Add final_answer tool
        tools_list.append(
            "- final_answer: Provide the final answer to the query\n"
            "  Takes inputs: {'answer': 'The final answer to the query'}\n"
            "  Returns an output of type: string"
        )
        
        return "\n".join(tools_list)
    
    def _parse_action(self, response):
        """Parse the action from the model response"""
        # Extract the action block using regular expressions
        action_match = re.search(r'Action:\s*\{(.*?)\}', response, re.DOTALL)
        if not action_match:
            self.logger.warning("No action found in response")
            return None
        
        # Try to extract the action details
        try:
            # Extract action name
            name_match = re.search(r'"name":\s*"([^"]+)"', action_match.group(0))
            if not name_match:
                self.logger.warning("No action name found in response")
                return None
            name = name_match.group(1)
            
            # Extract arguments
            args_match = re.search(r'"arguments":\s*\{(.*?)\}', action_match.group(0), re.DOTALL)
            arguments = {}
            
            if args_match:
                args_text = args_match.group(1).strip()
                # Parse individual arguments
                arg_matches = re.finditer(r'"([^"]+)":\s*"([^"]+)"', args_text)
                for match in arg_matches:
                    key, value = match.groups()
                    arguments[key] = value
            
            self.logger.info(f"Parsed action: {name} with arguments: {arguments}")
            return {"name": name, "arguments": arguments}
        except Exception as e:
            self.logger.error(f"Error parsing action details: {e}")
            
            # Fallback to a more basic parsing attempt for tools we know
            if "search_rag" in action_match.group(0):
                query_match = re.search(r'"query":\s*"([^"]+)"', action_match.group(0))
                if query_match:
                    self.logger.info(f"Fallback parsing: Using search_rag with query: {query_match.group(1)}")
                    return {
                        "name": "search_rag",
                        "arguments": {"query": query_match.group(1)}
                    }
            elif "search_web" in action_match.group(0):
                query_match = re.search(r'"query":\s*"([^"]+)"', action_match.group(0))
                if query_match:
                    self.logger.info(f"Fallback parsing: Using search_web with query: {query_match.group(1)}")
                    return {
                        "name": "search_web",
                        "arguments": {"query": query_match.group(1)}
                    }
            elif "read_paper" in action_match.group(0):
                url_match = re.search(r'"url":\s*"([^"]+)"', action_match.group(0))
                if url_match:
                    self.logger.info(f"Fallback parsing: Using read_paper with URL: {url_match.group(1)}")
                    return {
                        "name": "read_paper",
                        "arguments": {"url": url_match.group(1)}
                    }
            elif "final_answer" in action_match.group(0):
                answer_match = re.search(r'"answer":\s*"([^"]+)"', action_match.group(0))
                if answer_match:
                    self.logger.info("Fallback parsing: Using final_answer")
                    return {
                        "name": "final_answer",
                        "arguments": {"answer": answer_match.group(1)}
                    }
            
            # If all parsing attempts fail
            self.logger.warning("All parsing attempts failed")
            return None
    
    def _execute_action(self, action):
        """Execute the specified action"""
        name = action.get("name")
        arguments = action.get("arguments", {})
        
        if name == "final_answer":
            self.logger.info("Executing final_answer action")
            
            # Print a summary of all searches performed before the final answer
            if self.verbose:
                print("\n" + "="*60)
                print("SEARCH SUMMARY BEFORE FINAL ANSWER".center(60))
                print("="*60)
                if "search_rag" in self.used_tools:
                    print("✓ RESEARCH PAPERS were searched (RAG)")
                else:
                    print("✗ RESEARCH PAPERS were NOT searched (RAG)")
                    
                if "search_web" in self.used_tools:
                    print("✓ WEB SOURCES were searched (Bing)")
                else:
                    print("✗ WEB SOURCES were NOT searched (Bing)")
                    
                if "read_paper" in self.used_tools:
                    print("✓ PAPERS were downloaded and analyzed in detail")
                else:
                    print("✗ No papers were analyzed in detail")
                print("="*60 + "\n")
                
            return {"result": arguments.get("answer", "No answer provided"), "is_final": True}
        
        if name not in self.tools:
            self.logger.warning(f"Tool '{name}' not found")
            return {"result": f"Error: Tool '{name}' not found", "is_final": False}
        
        # Track which tools have been used
        self.used_tools.add(name)
        
        tool = self.tools[name]
        
        # Handle different argument types for different tools
        if name == "read_paper":
            url = arguments.get("url", "")
            # Get the research question from the current context
            research_question = self._get_research_question_from_context()
            self.logger.info(f"Executing {name} with URL: {url} and research question: {research_question}")
            result = tool.execute(url, research_question)
            formatted_result = tool.format_result(url, result)
        else:
            query = arguments.get("query", "")
            # Print detailed info for the user to see what's happening
            if self.verbose:
                if name == "search_rag":
                    print(f"\n[USING RAG SEARCH] Searching research papers for: {query}")
                    print("-" * 60)
                    print("This search looks through academic papers, research documents, and scientific literature.")
                    print("Results will include information from peer-reviewed sources and academic publications.")
                    print("-" * 60)
                elif name == "search_web":
                    print(f"\n[USING BING SEARCH] Searching the web for: {query}")
                    print("-" * 60)
                    print("This search looks through web pages, news articles, blogs, and other online sources.")
                    print("Results will include the most recent and relevant information from the internet.")
                    print("-" * 60)
                elif name == "search_arxiv":
                    print(f"\n[USING ARXIV SEARCH] Searching arxiv papers for: {query}")
                    print("-" * 60)
                    print("This search looks through academic papers on Arxiv.")
                    print("Results will include recent research papers and preprints.")
                    print("-" * 60)
            
            self.logger.info(f"Executing {name} with query: {query}")
            result = tool.execute(query)
            formatted_result = tool.format_result(query, result)
        
        return {"result": formatted_result, "is_final": False}
    
    def _get_research_question_from_context(self):
        """Extract the research question from the conversation context"""
        # First try to use the original query if available
        if self.original_query:
            return self.original_query
        
        # Otherwise try to find it in the context
        for message in self.context:
            if message["role"] == "user":
                return message["content"].strip()
        
        return "What are the main findings and implications of this research?"
    
    def run(self, query):
        """Run the ReAct agent on a query"""
        self.logger.info(f"Running agent with query: {query}")
        print(f"\nQuery: {query}")
        
        # Reset the used tools and context for this run
        self.used_tools = set()
        self.context = []
        self.original_query = query
        
        # Initialize conversation history with simple string replacement
        system_prompt = REACT_PROMPT.system_prompt.replace("{tools}", self.tools_description)
        
        # Add the task to the conversation
        initial_message = f"""
{query}

IMPORTANT INSTRUCTIONS:
You have to approach research like a human researcher collaborating with you:

1. You have to first reflect on your question to understand what you're asking and plan your approach.
2. You have main research tools:
   - search_rag: For searching internal documents and research papers
   - search_web: For searching public information on the internet
   - search_arxiv: For searching academic papers on Arxiv.org
   - read_paper: For downloading and reading academic papers
   - ask_user: Ask the user (supervisor) for feedback, clarification, or scope (don't use it unless you really need to)

3. For technical questions like "How can I quantify paraffin content in crude oil?", you have to check both internal resources and public information, asking clarifying questions when needed.

4. For factual questions like sports results, you have to primarily use web search and provide direct answers when available.

5. For company-specific questions like financial results, you have to prioritize internal documents while confirming with me if you need more context.

6. You have to think critically throughout the process - planning, analyzing, reconsidering approaches and ensuring you're addressing the needs effectively.

**ALWAYS CALL AN ACTION, don't forget about it.**
"""
        self.context.append({"role": "user", "content": initial_message})
        
        iteration = 0
        while iteration < config.MAX_ITERATIONS:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration}")
            print(f"\nIteration {iteration}----------------------------------")
            
            try:
                # Generate the next action
                self.logger.info("Generating model response")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *self.context
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                
                assistant_message = response.choices[0].message.content
                print(f"\nAssistant: {assistant_message}")
                self.context.append({"role": "assistant", "content": assistant_message})

                # Parse and execute the action
                action = self._parse_action(assistant_message)
                if not action:
                    self.logger.warning("Failed to parse action, asking for clarification")
                    self.context.append({"role": "user", "content": "I couldn't understand your action. Please provide a valid action in the format: Action: {\"name\": \"tool_name\", \"arguments\": {\"query\": \"your query\"}}."})
                    continue
                
                # Execute the action
                self.logger.info(f"Executing action: {action.get('name')}")
                result = self._execute_action(action)
                
                # If this is the final answer, check if we used both tools
                if result["is_final"]:
                    self.logger.info("Final answer received")
                    return result["result"]
                    
                # Format observation with "Observation:" prefix to match examples in prompts.py
                observation = f"Observation: {result['result']}"
                print(f"\nObservation: {observation}")
                self.context.append({"role": "user", "content": observation})
                self.logger.info("Added observation to context")
                
            except Exception as e:
                self.logger.error(f"Error during iteration {iteration}: {e}")
                return f"Error: {str(e)}"
        
        # If we reach the maximum number of iterations, return the last response
        self.logger.warning(f"Maximum iterations ({config.MAX_ITERATIONS}) reached without final answer")
        return "Maximum iterations reached without a final answer." 