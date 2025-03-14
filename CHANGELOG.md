# Changelog

All notable changes to the Weavr Alpha project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-03-14

### Added

- Structured Memory System to replace traditional RAG
- File monitoring system that updates structured knowledge when files change
- Knowledge organization by documents, sections, and themes
- Direct context injection into Gemini's context window

### Changed

- Switched from TogetherAI models to Google's Gemini Flash 2.0
- Updated system prompts and configuration for Gemini integration
- Improved knowledge retrieval with structured approach instead of vector search
- Updated launch script to use the new entry point

### Removed

- Traditional RAG system (FAISS, vector embeddings, BM25)
- Textual User Interface (TUI)
- Unused test files and directories
- Outdated dependencies (faiss-cpu, textual, rank_bm25)

### Known Issues

- Chain of Thought (CoT) functionality is currently broken due to the switch from TogetherAI models to Gemini
- Will be reimplemented in a future update

## [0.1.0] - 2024-02-24

### Added

- Initial release of Weavr Alpha
- RAG system using FAISS and TogetherAI embeddings
- Textual User Interface (TUI)
- Basic command system
- File change monitoring
- Chain of Thought reasoning
