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
    """Main function for running the DeepResearch agent"""
    parser = argparse.ArgumentParser(description="DeepResearch ReAct Agent")
    parser.add_argument("--query", type=str, help="Research query to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Use default query if none provided
    query = args.query or "What does it mean that RL generalizes?"
    
    # Print banner
    print("="*80)
    print("DeepResearch ReAct Agent".center(80))
    print("="*80)
    print(f"Query: {query}")
    print("-"*80)
    
    # Initialize and run the agent
    agent = ReActAgent(verbose=args.verbose)
    result = agent.run(query)
    
    # Print the result
    print("\n" + "="*80)
    print("FINAL ANSWER".center(80))
    print("="*80)
    print(result)
    print("="*80)

if __name__ == "__main__":
    main() 