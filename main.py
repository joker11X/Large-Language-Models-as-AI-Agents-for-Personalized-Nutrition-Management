from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import os
import tools
from typing import NoReturn
from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime
import threading
import sys
from claudeagent import ClaudeToolAgent
from app import run_server
from agent import console_agent, start_console_agent, get_available_agents, test_agents


def parse_arguments():
    if len(sys.argv) < 2:
        return None, "gpt4o-mini"
    
    mode = sys.argv[1]
    agent_type = "gpt4o-mini"  # 默认agent
    
    # 检查是否指定了agent类型
    if len(sys.argv) > 2:
        specified_agent = sys.argv[2]
        available_agents = get_available_agents()
        if specified_agent in available_agents:
            agent_type = specified_agent
        else:
            print(f"Warning: Unknown agent '{specified_agent}', using default 'gpt4o-mini'")
            print(f"Available agents: {', '.join(available_agents)}")
    
    return mode, agent_type


def normalize_agent(agent_type: str) -> str:
    alias = (agent_type or "").lower()
    mapping = {
        "openai": "dialog",
        "dialog": "dialog",
        "gemini": "vision",
        "vision": "vision",
        "claude": "file",
        "file": "file",
        "auto": "dialog",   # 推荐：web 用 dialog 启动，图片时后端自动切 vision
    }
    return mapping.get(alias, "dialog")


def show_menu():
    available_agents = get_available_agents()

    print("\n" + "="*60)
    print("AI Agent Runner — Commands")
    print("="*60)
    print("Format: <mode> [agent]")
    print()
    print("Modes:")
    print("  web         - start the web server (browser UI)")
    print("  web-clean   - start the web server and clear previous user data")
    print("  console     - interactive console with the agent")
    print("  both        - start web + console together")
    print("  test        - run connectivity/basic self-tests for all agents")
    print("  quit        - exit")
    print()
    print("Agents:")
    print("  gpt4o-mini  - OpenAI GPT-4o mini (fast, economical, tools-enabled)")
    print("  o1-mini     - OpenAI o1-mini (reasoning-optimized)")
    print("  gpt4o       - OpenAI GPT-4o (multimodal, tools-enabled)")
    print("  gemini      - Google Gemini 2.0 Flash (tools-enabled)")
    print("  claude      - Anthropic Claude Sonnet 4 (direct API)")
    print("  openai      - alias of gpt4o-mini (backward compatible)")
    print()
    print("Examples:")
    print("  web")
    print("    → Start web server with gpt4o-mini agent")
    print("  console o1-mini")
    print("    → Start console mode with o1-mini")
    print("  both gpt4o")
    print("    → Start web + console with gpt4o")
    print("  test")
    print("    → Run connectivity/basic-function tests for all agents")
    print()
    print(f"Available agents: {', '.join(available_agents)}")
    print("Default agent: gpt4o-mini")
    print("="*50)




def parse_command(command):
    parts = command.strip().split()
    if not parts:
        return None, "dialog"
    mode = parts[0].lower()
    agent_raw = parts[1] if len(parts) > 1 else "auto"
    agent_type = normalize_agent(agent_raw)

    valid_modes = ["web", "web-clean", "console", "both", "quit", "test", "selftest"]
    if mode not in valid_modes:
        return None, None
    return mode, agent_type


def execute_command(mode, agent_type, agent_type2=None):
    """Execute the selected command"""
    try:
        if mode == "web":
            print(f"Starting web server with {agent_type.upper()} agent.")
            print("Web UI: http://localhost:5000")
            print("Press Ctrl+C to stop the server")
            run_server(agent_type=agent_type)

        elif mode == "web-clean":
            print(f"Starting web server (clean data) with {agent_type.upper()} agent.")
            print("Web UI: http://localhost:5000")
            print("Press Ctrl+C to stop the server")
            run_server(clean_data=True, agent_type=agent_type)

        elif mode == "console":
            print(f"Starting console mode with {agent_type.upper()} agent.")
            print("Type 'quit' or 'exit' to leave console mode")
            console_agent(agent_type)

        elif mode == "both":
            print(f"Starting both web + console with {agent_type.upper()} agent.")
            print("Console commands are available in the background")
            print("Web UI: http://localhost:5000")
            print("Press Ctrl+C to stop all services")

            # start console agent thread
            start_console_agent(agent_type)

            # start web server
            run_server(agent_type=agent_type)

        elif mode == "test":
            print("Running self-tests for all available agents.")
            test_agents()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print("❌ Error while executing command:")
        print(f"Mode: {mode}, Agent: {agent_type}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("\nPlease check:")
        print("1) API keys are correctly set")
        print("2) Network connectivity is OK")
        print("3) Dependencies are properly installed")
        print("4) You have access to the requested model")


def main():
    """Main entry — Interactive mode"""
    print("Welcome to the AI Agent Test Runner!")
    print(f"Supported agents: {', '.join(get_available_agents())}")

    while True:
        try:
            show_menu()

            # user input
            user_input = input("Enter command: ").strip()
            if not user_input:
                continue

            # parse
            mode, agent_type = parse_command(user_input)
            if mode is None and agent_type is None:
                print("Invalid command. Please try again.")
                continue

            # special commands (help removed)
            if mode == "quit":
                print("Goodbye!")
                break

            # execute
            print(f"\nRunning: {mode} {agent_type}")
            execute_command(mode, agent_type)

            # return to menu after certain modes
            if mode in ["console", "test"]:
                print("\nBack to main menu...")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Back to main menu...")
            continue
        except EOFError:
            print("\n\nEOF detected. Exiting.")
            break


if __name__ == "__main__":
    main()