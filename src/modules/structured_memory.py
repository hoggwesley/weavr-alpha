import os
import yaml
import json
import re
from pathlib import Path
import time
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from threading import Event

class StructuredMemory:
    """
    StructuredMemory replaces the RAG system with a preloaded, structured knowledge store.
    Instead of performing real-time retrieval, it maintains persistent structured knowledge
    that can be directly injected into Gemini Flash's context window.
    """
    
    def __init__(self, knowledge_base_dir, storage_path=None):
        """
        Initialize the structured memory system.
        
        Args:
            knowledge_base_dir (str): Path to the knowledge base directory (e.g., Obsidian vault)
            storage_path (str): Path to store the structured memory files
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.storage_path = storage_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                       "data", "structured_memory")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Main storage for structured document knowledge
        self.knowledge_store = {}
        
        # File change handling
        self.update_flag = Event()
        self._observer = None
        
        # Load existing structured knowledge if available
        self._load_structured_knowledge()
    
    def _load_structured_knowledge(self):
        """Load structured knowledge from stored YAML files"""
        yaml_path = os.path.join(self.storage_path, "structured_knowledge.yaml")
        
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    self.knowledge_store = yaml.safe_load(f) or {}
                print(f"✅ Loaded structured knowledge from {yaml_path}")
            except Exception as e:
                print(f"❌ Error loading structured knowledge: {e}")
                self.knowledge_store = {}
        else:
            print("⚠️ No existing structured knowledge found. Building from scratch.")
            self._build_structured_knowledge()
    
    def _build_structured_knowledge(self):
        """
        Build structured knowledge from the documents in the knowledge base directory.
        This extracts and structures the content from markdown/text files.
        """
        print(f"🔹 Building structured knowledge from: {self.knowledge_base_dir}")
        self.knowledge_store = {}
        
        # Walk through the knowledge base directory
        for root, _, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    file_path = os.path.join(root, file)
                    try:
                        # Process the file and add to knowledge store
                        self._process_document(file_path)
                    except Exception as e:
                        print(f"❌ Error processing {file_path}: {e}")
        
        # Save the structured knowledge
        self._save_structured_knowledge()
    
    def _process_document(self, file_path):
        """
        Process a document and extract structured knowledge.
        
        Args:
            file_path (str): Path to the document file
        """
        # Get relative path for document ID
        rel_path = os.path.relpath(file_path, self.knowledge_base_dir)
        doc_id = rel_path.replace('\\', '/')
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get file modification time
        mod_time = os.path.getmtime(file_path)
        
        # Skip if the document hasn't changed since last processing
        if doc_id in self.knowledge_store and self.knowledge_store[doc_id].get('last_updated') == mod_time:
            return
        
        # Extract document structure
        title = os.path.basename(file_path).replace('.md', '').replace('.txt', '')
        sections = self._extract_sections(content)
        key_passages = self._extract_key_passages(content)
        themes = self._extract_themes(content)
        
        # Store structured knowledge
        self.knowledge_store[doc_id] = {
            'title': title,
            'sections': sections,
            'key_passages': key_passages,
            'themes': themes,
            'last_updated': mod_time,
            'file_path': file_path
        }
        
        print(f"✅ Processed document: {doc_id}")
    
    def _extract_sections(self, content):
        """
        Extract sections and subsections from document content.
        
        Args:
            content (str): Document content
            
        Returns:
            list: List of section dictionaries
        """
        sections = []
        
        # Split content by markdown headers
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Save previous section if exists
                if current_section:
                    sections.append({
                        'name': current_section,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_section = header_match.group(2)
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section:
            sections.append({
                'name': current_section,
                'content': '\n'.join(current_content).strip()
            })
        
        # If no sections were found, create a default section with the whole content
        if not sections:
            sections.append({
                'name': 'Main Content',
                'content': content.strip()
            })
        
        return sections
    
    def _extract_key_passages(self, content):
        """
        Extract key passages from document content.
        Key passages are identified by blockquotes or specific markers.
        
        Args:
            content (str): Document content
            
        Returns:
            list: List of key passage strings
        """
        key_passages = []
        
        # Extract blockquotes
        blockquote_pattern = r'^>\s*(.+?)$'
        for match in re.finditer(blockquote_pattern, content, re.MULTILINE):
            key_passages.append(match.group(1).strip())
        
        # Extract content between key passage markers (if used)
        marker_pattern = r'KEY PASSAGE START(.*?)KEY PASSAGE END'
        for match in re.finditer(marker_pattern, content, re.DOTALL):
            key_passages.append(match.group(1).strip())
        
        return key_passages
    
    def _extract_themes(self, content):
        """
        Extract themes from document content.
        Themes are extracted from tags or specific markers.
        
        Args:
            content (str): Document content
            
        Returns:
            list: List of theme strings
        """
        themes = []
        
        # Extract hashtags/tags
        tag_pattern = r'#([a-zA-Z0-9_-]+)'
        themes.extend([match.group(1) for match in re.finditer(tag_pattern, content)])
        
        return themes
    
    def _save_structured_knowledge(self):
        """Save the structured knowledge to a YAML file"""
        yaml_path = os.path.join(self.storage_path, "structured_knowledge.yaml")
        
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.knowledge_store, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"✅ Saved structured knowledge to {yaml_path}")
        except Exception as e:
            print(f"❌ Error saving structured knowledge: {e}")
    
    def get_knowledge_for_query(self, query):
        """
        Get structured knowledge relevant to a query.
        This doesn't do vector similarity search - it performs basic keyword matching
        to find the most relevant sections from the structured knowledge.
        
        Args:
            query (str): The user query
            
        Returns:
            str: Formatted structured knowledge for context injection
        """
        if not self.knowledge_store:
            return ""
        
        # Simple relevance scoring based on keyword overlap
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Score each document and its sections
        relevant_sections = []
        
        for doc_id, doc_data in self.knowledge_store.items():
            doc_title = doc_data['title']
            
            for section in doc_data['sections']:
                section_name = section['name']
                section_content = section['content']
                
                # Count keyword matches in section name and content
                section_text = (section_name + ' ' + section_content).lower()
                matches = sum(1 for term in query_terms if term in section_text)
                
                if matches > 0:
                    relevant_sections.append({
                        'doc_title': doc_title,
                        'section_name': section_name,
                        'content': section_content,
                        'score': matches,
                        'doc_id': doc_id
                    })
        
        # Sort by relevance score
        relevant_sections.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top sections (limit to avoid context overflow)
        top_sections = relevant_sections[:5]
        
        # Format the context for injection
        if not top_sections:
            return ""
        
        context_parts = ["### Relevant Knowledge from Your Personal Knowledge Base:"]
        
        for section in top_sections:
            context_parts.append(f"## {section['doc_title']} - {section['section_name']}")
            context_parts.append(section['content'])
            
            # Add key passages if available
            doc_id = section['doc_id']
            key_passages = self.knowledge_store[doc_id].get('key_passages', [])
            if key_passages:
                context_parts.append("\nKey Passages:")
                for passage in key_passages[:2]:  # Limit to avoid overwhelming
                    context_parts.append(f"> {passage}")
        
        return "\n\n".join(context_parts)
    
    def update_knowledge(self):
        """Force update of the structured knowledge"""
        self._build_structured_knowledge()
    
    def export_knowledge(self, format='yaml'):
        """
        Export the structured knowledge to a file.
        
        Args:
            format (str): Export format ('yaml' or 'json')
            
        Returns:
            str: Path to the exported file
        """
        export_path = os.path.join(self.storage_path, f"knowledge_export.{format}")
        
        try:
            if format.lower() == 'yaml':
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(self.knowledge_store, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            elif format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge_store, f, indent=2, ensure_ascii=False)
            else:
                return f"❌ Unsupported export format: {format}"
                
            return f"✅ Exported structured knowledge to {export_path}"
        except Exception as e:
            return f"❌ Error exporting structured knowledge: {e}"
    
    def start_file_watching(self):
        """
        Start watching for file changes in the knowledge base directory.
        When a file changes, the update flag is set to trigger knowledge update.
        """
        class StructuredMemoryFileHandler(FileSystemEventHandler):
            def __init__(self, update_flag):
                self.update_flag = update_flag
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(('.md', '.txt')):
                    print(f"🔹 File modified: {event.src_path}")
                    self.update_flag.set()
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith(('.md', '.txt')):
                    print(f"🔹 File created: {event.src_path}")
                    self.update_flag.set()
            
            def on_deleted(self, event):
                if not event.is_directory and event.src_path.endswith(('.md', '.txt')):
                    print(f"🔹 File deleted: {event.src_path}")
                    self.update_flag.set()
        
        event_handler = StructuredMemoryFileHandler(self.update_flag)
        self._observer = Observer()
        self._observer.schedule(event_handler, path=self.knowledge_base_dir, recursive=True)
        self._observer.start()
        print(f"✅ Started file change observer for {self.knowledge_base_dir}")
    
    def stop_file_watching(self):
        """Stop watching for file changes"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            print("✅ Stopped file change observer")
    
    def check_for_updates(self):
        """
        Check if the update flag is set and update knowledge accordingly.
        Returns True if knowledge was updated, False otherwise.
        """
        if self.update_flag.is_set():
            print("🔹 Update flag set, updating structured knowledge...")
            self.update_knowledge()
            self.update_flag.clear()
            return True
        return False


# Create a singleton instance for global access
structured_memory = None

def initialize_structured_memory(knowledge_base_dir):
    """Initialize the structured memory singleton"""
    global structured_memory
    structured_memory = StructuredMemory(knowledge_base_dir)
    structured_memory.start_file_watching()
    return structured_memory

def get_structured_memory():
    """Get the structured memory singleton instance"""
    global structured_memory
    if not structured_memory:
        raise ValueError("Structured memory not initialized. Call initialize_structured_memory first.")
    return structured_memory