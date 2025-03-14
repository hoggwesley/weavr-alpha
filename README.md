# Weavr Alpha

Weavr Alpha is an AI-powered Obsidian companion that enhances your knowledge management experience by providing intelligent interactions with your personal knowledge base.

## Overview

Weavr Alpha connects to your Obsidian vault and uses advanced AI models (currently Google's Gemini) to help you interact with, query, and expand your knowledge base. Unlike traditional AI assistants, Weavr prioritizes your personal knowledge and can provide responses informed by your own notes and documents.

## Current Architecture

### Core Components

#### Structured Memory System

- **Smart Knowledge Organization**: Automatically extracts and organizes information from your Markdown files into a structured format
- **Document Analysis**: Identifies sections, key passages, and themes in your documents
- **Persistent Storage**: Maintains structured knowledge between sessions for faster startup
- **File Monitoring**: Watches for changes in your knowledge base and updates the structured memory in real-time
- **Intelligent File Prioritization**: Prioritizes files based on directory depth, keywords, and recency when file limits are reached
- **Knowledge Context Management**: Allows clearing knowledge context without disabling the system

#### AI Generation

- **Gemini Integration**: Uses Google's Gemini Flash 2.0 as the primary language model
- **Context Injection**: Directly injects relevant structured knowledge into the model's context window
- **Configurable System Prompts**: Allows customization of AI behavior through editable system prompts
- **Document Title Awareness**: Enhanced recognition of document titles in queries for more relevant responses

#### Configuration System

- **YAML-based Config**: All settings stored in `config.yaml` for easy modification
- **API Key Management**: Securely manages API keys for various services
- **Knowledge Base Settings**: Configurable paths and processing options for your Obsidian vault

### Command Interface

- **Command-based Interaction**: Provides various commands for managing the system, viewing memory, and controlling AI behavior
- **Memory Management**: Commands for viewing and manipulating the structured knowledge
- **Conversation History**: Maintains context across multiple interactions

## Known Issues

### Chain of Thought (CoT)

- The CoT functionality is currently broken due to the switch from TogetherAI models to Gemini
- We plan to implement a new version of CoT specifically designed for Gemini models in a future update

## Future Development Plans

### Planned Enhancements

- **New CoT Implementation**: Developing a Chain of Thought reasoning system compatible with Gemini models
- **Improved Knowledge Structuring**: Enhancing the structured memory system with better semantic understanding
- **New User Interface**: Designing a more intuitive and responsive interface for interacting with Weavr
- **Multi-Modal Support**: Adding capabilities to handle images and other media types within your knowledge base
- **Enhanced Memory Management**: Implementing more sophisticated memory management and context handling

### Technical Roadmap

- Improve documentation and code organization
- Enhance error handling and stability
- Optimize performance for larger knowledge bases
- Add unit tests for core functionality

## Getting Started

### Prerequisites

- Python 3.8 or higher
- An Obsidian vault with your knowledge base
- A Google API key for Gemini access

### Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your settings in `config.yaml`
4. Run the application using `launch_weavr.bat` or `python src/scripts/run_weavr.py`

### Configuration

Edit the `config.yaml` file to set:

- Path to your Obsidian vault
- API key for Gemini
- System prompt and behavior settings
- Debug and logging options

## Usage

### Basic Commands

- Start a conversation by typing your query
- Use `/knowledge` to manage your structured knowledge system
- Use `/knowledge status` to view knowledge base structure
- Use `/knowledge set` to set knowledge base directory
- Use `/knowledge limit` to set maximum files to process
- Use `/knowledge clear` to clear knowledge context and optionally set a new directory
- Use `/exit` to quit the application

## Changelog

See the [CHANGELOG.md](CHANGELOG.md) file for a detailed history of changes to the project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the LICENSE file included in the repository.
