# AgentQ - Advanced Reasoning and Learning for Autonomous AI Agents
GROUP PROJECT FOR CMPS 6740 Reinforcement Learning

AgentQ is a sophisticated AI framework designed to enhance autonomous reasoning and learning capabilities for AI agents. It leverages various agentic architectures to perform complex tasks on the web reliably.

This repository is built upon the open-source implementation of the research paper [Agent Q](https://arxiv.org/abs/2408.07199), expanding its functionalities to explore reinforcement learning (RL) and fine-tuning strategies specifically designed for interactive and dynamic environments. By augmenting the existing framework with additional RL-based components, this project aims to push the boundaries of what autonomous agents can achieve in partially observable environments.

## Features

AgentQ includes the following architectures:
- Planner <> Navigator multi-agent architecture
- Solo planner-actor agent
- Actor <> Critic multi-agent architecture
- Actor <> Critic architecture with Monte Carlo Tree Search-based reinforcement learning and DPO fine-tuning
- Vision Agent: Implemented but not fully integrated, with potential for expanding multimodal capabilities in future iterations.


This repository enhances the original implementation by introducing new features and experimental configurations, particularly for RL-based workflows.

## OpenAI Integration

AgentQ leverages OpenAI's advanced language models to enhance its decision-making and interaction capabilities. This integration provides:

- **Enhanced Prompting**: Utilizes OpenAI's models to generate more accurate and context-aware responses, improving the agent's ability to understand and execute complex tasks.
- **Real-time API Access**: Connects to OpenAI's API for real-time processing,

### Setting Up OpenAI API

1. **Obtain an API Key**: Sign up for an OpenAI account and obtain an API key from the [OpenAI Dashboard](https://platform.openai.com/).
2. **Configure Environment Variables**:
   - Add your OpenAI API key to the `.env` file:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Setup

### Prerequisites

1. **Python**: Ensure you have Python 3.10 or later installed.
2. **Poetry**: We recommend using Poetry for dependency management. Install it using the [official instructions](https://python-poetry.org/docs/#installation).

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/agentq.git
   cd agentq


2. **Install Dependencies**:
   Use Poetry to install the required dependencies.
   ```bash
   poetry install
   ```

3. **Configure Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your OpenAI and Langfuse API keys. Refer to `.env.example` for guidance.

4. **Start Chrome in Dev Mode**:
   - **Mac**:
     ```bash
     sudo /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
     ```
   - **Linux**:
     ```bash
     google-chrome --remote-debugging-port=9222
     ```
   - **Windows**:
     ```bash
     "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
     ```

### Running the Agent

To run the agent, execute the following command:

```bash
python -u -m agentq
```

### run evals

```bash
 python -m test.tests_processor --orchestrator_type fsm
```

### generate dpo pairs for RL

```bash
python -m agentq.core.mcts.browser_mcts
```

