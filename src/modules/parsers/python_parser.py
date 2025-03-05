import ast
import os
import re
from typing import Dict, List, Optional, Tuple

class PythonFileParser:
    """Parser for extracting structured information from Python files"""
    
    def __init__(self):
        self.function_details = {}
        self.class_details = {}
        self.imports = []
        self.comments = []
        self.docstrings = {}
    
    def parse_file(self, file_path: str) -> Dict:
        """Parse a Python file and extract its structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Extract comments that aren't part of docstrings
            self.extract_comments(content)
            
            # Parse the Python code
            tree = ast.parse(content)
            self.extract_imports(tree)
            self.extract_functions_and_classes(tree)
            
            # Create structured representation
            filename = os.path.basename(file_path)
            result = {
                "filename": filename,
                "path": file_path,
                "type": "python_file",
                "imports": self.imports,
                "functions": self.function_details,
                "classes": self.class_details,
                "comments": self.comments,
                "docstrings": self.docstrings
            }
            
            # Add a summary for easy retrieval
            result["summary"] = self._generate_summary(result)
            
            return result
        except Exception as e:
            return {
                "filename": os.path.basename(file_path),
                "path": file_path,
                "type": "python_file",
                "error": str(e),
                "summary": f"Error parsing Python file: {str(e)}"
            }
    
    def extract_comments(self, content: str):
        """Extract comments from Python code"""
        # Simple regex to find comments (this is not perfect but works for basic cases)
        comment_pattern = r'^\s*#\s*(.*)$'
        self.comments = [match.group(1) for match in re.finditer(comment_pattern, content, re.MULTILINE)]
    
    def extract_imports(self, tree):
        """Extract import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    self.imports.append(f"from {module} import {name.name}")
    
    def extract_functions_and_classes(self, tree):
        """Extract function and class definitions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._process_function(node)
            elif isinstance(node, ast.ClassDef):
                self._process_class(node)
    
    def _process_function(self, node):
        """Process a function definition"""
        func_name = node.name
        args = [arg.arg for arg in node.args.args]
        docstring = ast.get_docstring(node)
        
        self.function_details[func_name] = {
            "name": func_name,
            "arguments": args,
            "line_number": node.lineno,
            "has_docstring": docstring is not None
        }
        
        if docstring:
            self.docstrings[f"function:{func_name}"] = docstring
    
    def _process_class(self, node):
        """Process a class definition"""
        class_name = node.name
        bases = [self._get_name(base) for base in node.bases]
        docstring = ast.get_docstring(node)
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name
                methods.append(method_name)
                method_docstring = ast.get_docstring(item)
                if method_docstring:
                    self.docstrings[f"method:{class_name}.{method_name}"] = method_docstring
        
        self.class_details[class_name] = {
            "name": class_name,
            "bases": bases,
            "methods": methods,
            "line_number": node.lineno,
            "has_docstring": docstring is not None
        }
        
        if docstring:
            self.docstrings[f"class:{class_name}"] = docstring
    
    def _get_name(self, node):
        """Helper function to get names from AST nodes"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return "unknown"
    
    def _generate_summary(self, data: Dict) -> str:
        """Generate a summary of the Python file for RAG retrieval"""
        filename = data["filename"]
        summary_parts = [f"Python file: {filename}"]
        
        # Add classes
        if data["classes"]:
            class_names = list(data["classes"].keys())
            summary_parts.append(f"Classes: {', '.join(class_names)}")
        
        # Add functions
        if data["functions"]:
            function_names = list(data["functions"].keys())
            summary_parts.append(f"Functions: {', '.join(function_names)}")
        
        # Add imports summary
        if data["imports"]:
            modules = []
            for imp in data["imports"]:
                if imp.startswith("import "):
                    modules.append(imp[7:])
                elif imp.startswith("from "):
                    parts = imp.split(" import ")
                    if len(parts) > 1:
                        modules.append(f"{parts[0][5:]}.{parts[1]}")
            
            if modules:
                summary_parts.append(f"Uses modules: {', '.join(modules[:5])}")
                if len(modules) > 5:
                    summary_parts[-1] += f" and {len(modules)-5} more"
        
        return " | ".join(summary_parts)


def parse_python_file(file_path: str) -> Dict:
    """Parse a Python file and return structured information"""
    parser = PythonFileParser()
    return parser.parse_file(file_path)
