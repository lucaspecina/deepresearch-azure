"""
DeepResearch with ReAct main script.
Implements a ReAct-based agent for research queries using RAG and web search.
"""

import argparse
import logging
from deepresearch_azure.react_agent import ReActAgent

# Configure logging
def setup_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('deepresearch')

def main():
    """Main function for running the DeepResearch agent interactively."""
    parser = argparse.ArgumentParser(description="DeepResearch ReAct Agent (interactive)")
    parser.add_argument("--query", type=str, help="Initial research query to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)

    # Initialize the agent once
    agent = ReActAgent(verbose=args.verbose)

    # Welcome banner
    print("\n" + "="*80)
    print("DeepResearch ReAct Agent".center(80))
    print("="*80)

    # Start interactive loop
    next_query = args.query
    while True:
        # Prompt for query if none
        if not next_query:
            next_query = input("\nEnter a research query (or 'exit' to quit): ").strip()
        # Exit condition
        if not next_query or next_query.lower() in ("exit", "quit"):
            print("\nThank you for using DeepResearch. Goodbye!")
            break

        # Display the query
        print("\n" + "-"*80)
        print(f"Query: {next_query}")
        print("-"*80)

        # Run the agent
        result = agent.run(next_query)

        # Show the result
        print("\n" + "="*80)
        print("RESULT".center(80))
        print("="*80)
        print(result)
        print("="*80)

        # Analysis summary
        print("\n" + "-"*80)
        print("ANALYSIS SUMMARY".center(80))
        print("-"*80)
        if agent.used_tools:
            if 'search_rag' in agent.used_tools:
                print("✓ Performed internal documentation search.")
            else:
                print("✗ No internal docs search performed.")
            if 'search_web' in agent.used_tools:
                print("✓ Performed web search.")
            else:
                print("✗ No web search performed.")
            if 'ask_user' in agent.used_tools:
                print("✓ Asked clarifying questions to the user.")
            else:
                print("✗ No clarifying questions were asked.")
        else:
            print("No tools were used in this session.")
        print("-"*80)

        # Suggestions to continue the conversation
        print("\nSuggestions:")
        print("- Ask for more methodology details")
        print("- Challenge or refine the conclusion")
        print("- Ask a follow-up question on this topic")
        print("- Start a new topic")

        # Next action prompt
        print("\nWhat would you like to do next? You can ask a follow-up, start a new query, or type 'exit'.")
        next_query = input("> ").strip()

if __name__ == "__main__":
    main() 