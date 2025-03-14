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
    StructuredMemory provides a preloaded, structured knowledge store for Weavr AI.
    Instead of performing real-time retrieval, it maintains persistent structured knowledge
    that can be directly injected into Gemini Flash's context window.
    """
    
    def __init__(self, knowledge_base_dir, storage_path=None, file_limit=100):
        """
        Initialize the structured memory system.
        
        Args:
            knowledge_base_dir (str): Path to the knowledge base directory (e.g., Obsidian vault)
            storage_path (str): Path to store the structured memory files
            file_limit (int): Maximum number of files to process (default: 100)
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.storage_path = storage_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                       "data", "structured_memory")
        self.file_limit = file_limit
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Main storage for structured document knowledge
        self.knowledge_store = {}
        
        # Store information about excluded files
        self.excluded_files = []
        self.total_eligible_files = 0
        
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
                    loaded_data = yaml.safe_load(f) or {}
                
                # Check if the loaded knowledge is from the current knowledge base directory
                # Look at the file_path of the first document to determine the source directory
                if loaded_data:
                    first_doc = next(iter(loaded_data.values()))
                    if 'file_path' in first_doc:
                        stored_path = os.path.dirname(first_doc['file_path'])
                        if not os.path.normpath(stored_path).startswith(os.path.normpath(self.knowledge_base_dir)):
                            print(f"⚠️ Loaded knowledge is from a different directory. Rebuilding from scratch.")
                            self._build_structured_knowledge()
                            return
                
                # Count total eligible files in the directory
                total_eligible_files = 0
                eligible_files = []
                for root, _, files in os.walk(self.knowledge_base_dir):
                    for file in files:
                        if file.endswith(('.md', '.txt')):
                            file_path = os.path.join(root, file)
                            try:
                                mod_time = os.path.getmtime(file_path)
                                eligible_files.append((file_path, mod_time))
                                total_eligible_files += 1
                            except Exception:
                                pass
                
                self.total_eligible_files = total_eligible_files
                
                # Apply prioritization algorithm to eligible files
                eligible_files = self._prioritize_files(eligible_files)
                
                # Apply file limit to the loaded data
                if len(loaded_data) > self.file_limit:
                    print(f"⚠️ Loaded knowledge contains {len(loaded_data)} files, limiting to {self.file_limit} most recently modified files")
                    
                    # Get the file paths from loaded data
                    loaded_files = [(doc.get('file_path', ''), doc.get('last_updated', 0)) 
                                   for doc in loaded_data.values()]
                    
                    # Apply prioritization algorithm
                    loaded_files = self._prioritize_files(loaded_files)
                    
                    # Keep only the highest priority files within the limit
                    keep_paths = set(path for path, _ in loaded_files[:self.file_limit])
                    
                    # Filter the knowledge store
                    self.knowledge_store = {
                        doc_id: doc for doc_id, doc in loaded_data.items()
                        if doc.get('file_path', '') in keep_paths
                    }
                    
                    # Determine excluded files
                    known_paths = set(doc.get('file_path', '') for doc in self.knowledge_store.values())
                    self.excluded_files = [
                        os.path.relpath(path, self.knowledge_base_dir).replace('\\', '/')
                        for path, _ in eligible_files
                        if path not in known_paths
                    ]
                else:
                    self.knowledge_store = loaded_data
                    
                    # Determine excluded files
                    if total_eligible_files > len(self.knowledge_store):
                        known_paths = set(doc.get('file_path', '') for doc in self.knowledge_store.values())
                        self.excluded_files = [
                            os.path.relpath(path, self.knowledge_base_dir).replace('\\', '/')
                            for path, _ in eligible_files
                            if path not in known_paths
                        ]
                    else:
                        self.excluded_files = []
                
                print(f"✅ Loaded structured knowledge from {yaml_path}")
            except Exception as e:
                print(f"❌ Error loading structured knowledge: {e}")
                self.knowledge_store = {}
                self._build_structured_knowledge()
        else:
            print("⚠️ No existing structured knowledge found. Building from scratch.")
            self._build_structured_knowledge()
    
    def _prioritize_files(self, file_list):
        """
        Prioritize files based on directory depth and special keywords.
        
        Args:
            file_list: List of tuples (file_path, modification_time)
            
        Returns:
            Sorted list of tuples with highest priority files first
        """
        # Calculate priority scores for each file
        scored_files = []
        
        for file_path, mod_time in file_list:
            # Start with base score (higher is better)
            score = 1000
            
            # Get relative path from knowledge base directory
            rel_path = os.path.relpath(file_path, self.knowledge_base_dir)
            
            # 1. Prioritize files in parent directory (fewer path separators)
            depth = rel_path.count(os.sep) + rel_path.count('/')
            score -= depth * 50  # Penalize deeper files
            
            # 2. Deprioritize files with special keywords
            lower_path = rel_path.lower()
            if any(keyword in lower_path for keyword in ['legacy', 'retired', 'ignore', 'archive']):
                score -= 500  # Significant penalty for deprecated content
            
            # 3. Boost files with important keywords
            if any(keyword in lower_path for keyword in ['important', 'key', 'main', 'index']):
                score += 200
                
            # 4. Recent files still matter, so include mod time as a factor
            # Convert mod_time to days since epoch for a reasonable scale
            days_since_epoch = mod_time / (60 * 60 * 24)
            score += days_since_epoch * 0.1  # Small boost for recency
            
            scored_files.append((file_path, mod_time, score))
        
        # Sort by score (descending) and return original tuples
        scored_files.sort(key=lambda x: x[2], reverse=True)
        return [(path, mod_time) for path, mod_time, _ in scored_files]
    
    def _build_structured_knowledge(self):
        """
        Build structured knowledge from the documents in the knowledge base directory.
        This extracts and structures the content from markdown/text files.
        """
        print(f"🔹 Building structured knowledge from: {self.knowledge_base_dir}")
        self.knowledge_store = {}
        self.excluded_files = []
        
        # First, collect all eligible files with their modification times
        eligible_files = []
        for root, _, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    file_path = os.path.join(root, file)
                    try:
                        # Get file modification time
                        mod_time = os.path.getmtime(file_path)
                        eligible_files.append((file_path, mod_time))
                    except Exception as e:
                        print(f"❌ Error accessing {file_path}: {e}")
        
        # Store total eligible files count
        self.total_eligible_files = len(eligible_files)
        
        # Apply prioritization algorithm
        eligible_files = self._prioritize_files(eligible_files)
        
        # Apply file limit
        if len(eligible_files) > self.file_limit:
            print(f"⚠️ Directory contains {len(eligible_files)} files, limiting to {self.file_limit} highest priority files")
            # Store excluded files
            self.excluded_files = [os.path.relpath(path, self.knowledge_base_dir).replace('\\', '/') 
                                  for path, _ in eligible_files[self.file_limit:]]
            eligible_files = eligible_files[:self.file_limit]
        
        # Process the files
        for file_path, _ in eligible_files:
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
            
            # Check if query directly mentions the document title
            title_match = False
            title_words = set(re.findall(r'\b\w+\b', doc_title.lower()))
            title_overlap = len(query_terms.intersection(title_words))
            if title_overlap > 0 or any(term in query.lower() for term in [doc_title.lower()]):
                title_match = True
                
            for section in doc_data['sections']:
                section_name = section['name']
                section_content = section['content']
                
                # Count keyword matches in section name and content
                section_text = (section_name + ' ' + section_content).lower()
                matches = sum(1 for term in query_terms if term in section_text)
                
                # Boost score if the document title is directly mentioned in the query
                score = matches
                if title_match:
                    score += 5  # Significant boost for title matches
                
                if matches > 0 or title_match:
                    relevant_sections.append({
                        'doc_title': doc_title,
                        'section_name': section_name,
                        'content': section_content,
                        'score': score,
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
        
        # Group sections by document for better organization
        doc_sections = {}
        for section in top_sections:
            doc_title = section['doc_title']
            if doc_title not in doc_sections:
                doc_sections[doc_title] = []
            doc_sections[doc_title].append(section)
        
        # Format each document's sections
        for doc_title, sections in doc_sections.items():
            context_parts.append(f"## Document: {doc_title}")
            
            for section in sections:
                context_parts.append(f"### {section['section_name']}")
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
    
    def needs_update(self):
        """Check if the knowledge base needs to be updated"""
        return self.update_flag.is_set()
    
    def clear_knowledge(self):
        """
        Clear all knowledge from memory while keeping the system enabled.
        This allows toggling knowledge context on/off without reloading.
        """
        # Clear the knowledge store
        self.knowledge_store = {}
        self.excluded_files = []
        
        # Keep file watching active
        print("✅ Knowledge context cleared from memory")
        
        return True


# Create a singleton instance for global access
structured_memory = None

def initialize_structured_memory(knowledge_base_dir, file_limit=100):
    """Initialize the structured memory singleton"""
    global structured_memory
    
    # If we already have a structured memory instance, stop file watching
    if structured_memory:
        structured_memory.stop_file_watching()
        
        # Delete the existing structured knowledge YAML file to force a rebuild
        yaml_path = os.path.join(structured_memory.storage_path, "structured_knowledge.yaml")
        if os.path.exists(yaml_path):
            try:
                os.remove(yaml_path)
                print(f"✅ Deleted existing structured knowledge file to ensure clean rebuild")
            except Exception as e:
                print(f"⚠️ Could not delete existing structured knowledge file: {e}")
    
    # Create a new structured memory instance
    structured_memory = StructuredMemory(knowledge_base_dir, file_limit=file_limit)
    structured_memory.start_file_watching()
    return structured_memory

def get_structured_memory():
    """Get the structured memory singleton instance"""
    global structured_memory
    if not structured_memory:
        raise ValueError("Structured memory not initialized. Call initialize_structured_memory first.")
    return structured_memory