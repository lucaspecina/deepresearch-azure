"""
ReAct agent implementation for DeepResearch.
Uses a reasoning-action-observation cycle to solve tasks.
"""

import re
import json
import logging
from openai import AzureOpenAI
import deepresearch_azure.config as config
from deepresearch_azure.prompts import REACT_PROMPT
from deepresearch_azure.search_tools import get_all_tools

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
                print("="*60 + "\n")
                
            return {"result": arguments.get("answer", "No answer provided"), "is_final": True}
        
        if name not in self.tools:
            self.logger.warning(f"Tool '{name}' not found")
            return {"result": f"Error: Tool '{name}' not found", "is_final": False}
        
        # Track which tools have been used
        self.used_tools.add(name)
        
        tool = self.tools[name]
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
        
        self.logger.info(f"Executing {name} with query: {query}")
        result = tool.execute(query)
        formatted_result = tool.format_result(query, result)
        
        return {"result": formatted_result, "is_final": False}
    
    def run(self, query):
        """Run the ReAct agent on a query"""
        self.logger.info(f"Running agent with query: {query}")
        print(f"\nQuery: {query}")
        
        # Reset the used tools for this run
        self.used_tools = set()
        
        # Initialize conversation history with simple string replacement
        # Use string replacement instead of format to avoid issues with JSON braces
        system_prompt = REACT_PROMPT.system_prompt.replace("{tools}", self.tools_description)
        context = []
        
        # Add the task to the conversation
        initial_message = f"""
{query}

IMPORTANT INSTRUCTIONS:
You have to approach research like a human researcher collaborating with you:

1. You have to first reflect on your question to understand what you're asking and plan your approach.
2. You have two main research tools:
   - search_rag: For searching internal documents and research papers
   - search_web: For searching public information on the internet

3. For technical questions like "How can I quantify paraffin content in crude oil?", you have to check both internal resources and public information, asking clarifying questions when needed.

4. For factual questions like sports results, you have to primarily use web search and provide direct answers when available.

5. For company-specific questions like financial results, you have to prioritize internal documents while confirming with me if you need more context.

6. You have to think critically throughout the process - planning, analyzing, reconsidering approaches and ensuring you're addressing the needs effectively.
"""
        context.append({"role": "user", "content": initial_message})
        # context.append({"role": "user", "content": query})
        
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
                        *context
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                
                assistant_message = response.choices[0].message.content
                print(f"\nAssistant: {assistant_message}")
                context.append({"role": "assistant", "content": assistant_message})
                
                # # Print agent's thought process
                # if self.verbose:
                #     print("\nAgent thought process:")
                #     # Extract and print the Thought part if it exists
                #     thought_match = re.search(r'Thought:(.*?)(?:Action:|$)', assistant_message, re.DOTALL)
                #     if thought_match:
                #         thought = thought_match.group(1).strip()
                #         print(f"{thought}")
                #     print("\nAgent action:")
                # else:
                #     print(f"\nAssistant: {assistant_message}")
                
                # Parse and execute the action
                action = self._parse_action(assistant_message)
                # print(f"\nAction: {action}")
                if not action:
                    self.logger.warning("Failed to parse action, asking for clarification")
                    context.append({"role": "user", "content": "I couldn't understand your action. Please provide a valid action in the format: Action: {\"name\": \"tool_name\", \"arguments\": {\"query\": \"your query\"}}."})
                    continue
                
                # # If trying to provide final answer too early
                # if action.get("name") == "final_answer" and (
                #     iteration < self.min_iterations or 
                #     len(self.used_tools) < 2  # At least 2 different tools should be used
                # ):
                #     # Encourage using both search tools
                #     unused_tools = set(["search_rag", "search_web"]) - self.used_tools
                #     if unused_tools:
                #         suggested_tool = list(unused_tools)[0]
                #         tool_desc = "research papers" if suggested_tool == "search_rag" else "the web"
                #         self.logger.info(f"Encouraging use of {suggested_tool} before final answer")
                        
                #         # More detailed instruction when suggesting a search
                #         if self.verbose:
                #             print(f"\n[ALERT] Need to search {tool_desc} before finalizing answer!")
                            
                #         context.append({
                #             "role": "user", 
                #             "content": f"Please search {tool_desc} using {suggested_tool} before providing a final answer. We need information from both research papers AND web sources to give the most comprehensive answer."
                #         })
                #         continue
                
                # Execute the action
                self.logger.info(f"Executing action: {action.get('name')}")
                result = self._execute_action(action)
                
                # If this is the final answer, check if we used both tools
                if result["is_final"]:
                    # if len(self.used_tools) < 2 and iteration < config.MAX_ITERATIONS - 1:
                    #     unused_tools = set(["search_rag", "search_web"]) - self.used_tools
                    #     if unused_tools:
                    #         suggested_tool = list(unused_tools)[0]
                    #         tool_desc = "research papers" if suggested_tool == "search_rag" else "the web"
                    #         self.logger.info(f"Suggesting use of {suggested_tool} before finalizing")
                            
                    #         if self.verbose:
                    #             print(f"\n[ALERT] Need to search {tool_desc} before finalizing answer!")
                                
                    #         context.append({
                    #             "role": "user", 
                    #             "content": f"Before finalizing your answer, please also search {tool_desc} using {suggested_tool}. We need information from both research papers AND web sources."
                    #         })
                    #         continue
                    
                    self.logger.info("Final answer received")
                    # print(f"\nFinal answer: {result['result']}")
                    return result["result"]
                
                # Add the observation to the conversation
                # Add a preview of the results in verbose mode
                # if self.verbose:
                #     print("\n[OBSERVATION SUMMARY]")
                #     # Create a preview of the observation (first 150 chars)
                #     result_text = result["result"]
                #     preview = result_text[:150] + "..." if len(result_text) > 150 else result_text
                #     print(f"Received: {preview}")
                    
                # Format observation with "Observation:" prefix to match examples in prompts.py
                observation = f"Observation: {result['result']}"
                print(f"\nObservation: {observation}")
                context.append({"role": "user", "content": observation})
                self.logger.info("Added observation to context")
                
            except Exception as e:
                self.logger.error(f"Error during iteration {iteration}: {e}")
                return f"Error: {str(e)}"
        
        # If we reach the maximum number of iterations, return the last response
        self.logger.warning(f"Maximum iterations ({config.MAX_ITERATIONS}) reached without final answer")
        return "Maximum iterations reached without a final answer." 