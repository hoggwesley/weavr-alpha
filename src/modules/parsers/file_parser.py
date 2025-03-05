import os
import mimetypes
from typing import Dict, List, Optional
import re

# Import individual parsers
from modules.parsers.python_parser import parse_python_file

# Initialize MIME types
mimetypes.init()

def is_binary_file(file_path: str) -> bool:
    """Determine if a file is binary based on MIME type and content sampling"""
    # Check file extension first
    mime_type, _ = mimetypes.guess_type(file_path)
    if (mime_type and not mime_type.startswith('text/')):
        return True
        
    # Backup method: check file content
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            # If there are null bytes, it's likely binary
            if b'\x00' in chunk:
                return True
            # If more than 30% of the characters are non-ASCII, probably binary
            non_ascii = sum(1 for b in chunk if b > 127)
            if non_ascii > 0.3 * len(chunk):
                return True
        return False
    except Exception:
        # When in doubt, assume binary
        return True

def parse_file(file_path: str) -> Dict:
    """
    Parse a file based on its extension and return structured information.
    Returns a dictionary with file information.
    """
    if not os.path.exists(file_path):
        return {
            "filename": os.path.basename(file_path),
            "path": file_path,
            "type": "unknown",
            "error": "File not found",
            "content": ""
        }
    
    # Skip binary files immediately
    if is_binary_file(file_path):
        return {
            "filename": os.path.basename(file_path),
            "path": file_path,
            "type": "binary",
            "error": "Binary file not parsed",
            "content": ""
        }
        
    file_extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    # Special handling for model files in your project
    filename_lower = filename.lower()
    if "mixtral" in filename_lower or "mistral" in filename_lower or "_8x7b_" in filename_lower or "_7b_" in filename_lower:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract model details from the file
            model_type = ""
            if "mixtral" in filename_lower or "_8x7b_" in filename_lower:
                model_type = "mixtral"
            elif "mistral" in filename_lower or "_7b_" in filename_lower:
                model_type = "mistral"
            
            # Look for API name in the content
            api_name = None
            model_size = None
            
            if "mistralai/Mixtral-8x7B" in content:
                api_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                model_size = "8x7B"
            elif "mistralai/Mistral-7B" in content:
                api_name = "mistralai/Mistral-7B-Instruct-v0.1"
                model_size = "7B"
            
            # Create a more descriptive summary that includes functions
            functions = []
            class_pattern = re.compile(r'class\s+(\w+)')
            function_pattern = re.compile(r'def\s+(\w+)\s*\(')
            
            classes = class_pattern.findall(content)
            function_matches = function_pattern.findall(content)
            
            summary = f"AI model file: {model_type} ({model_size if model_size else 'unknown size'})"
            if classes:
                summary += f" | Classes: {', '.join(classes)}"
            if function_matches:
                functions = [f for f in function_matches if f not in ('__init__')]
                if functions:
                    summary += f" | Functions: {', '.join(functions)}"
            
            return {
                "filename": filename,
                "path": file_path,
                "type": "model_file",
                "model_type": model_type,
                "api_name": api_name,
                "model_size": model_size,
                "functions": functions,
                "content": content,
                "summary": summary
            }
        except UnicodeDecodeError:
            return {
                "filename": filename,
                "path": file_path,
                "type": "binary",
                "error": "Binary file not parsed",
                "content": ""
            }
        except Exception as e:
            return {
                "filename": filename,
                "path": file_path,
                "type": "unknown",
                "error": str(e),
                "content": ""
            }
    
    # Python files
    if file_extension == '.py':
        return parse_python_file(file_path)
    
    # Default text-based handling for other file types
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {
            "filename": filename,
            "path": file_path,
            "type": "text",
            "content": content
        }
    except UnicodeDecodeError:
        return {
            "filename": filename,
            "path": file_path,
            "type": "binary",
            "error": "Binary file not parsed (Unicode decode error)",
            "content": ""
        }
    except Exception as e:
        return {
            "filename": filename,
            "path": file_path,
            "type": "unknown",
            "error": str(e),
            "content": ""
        }
