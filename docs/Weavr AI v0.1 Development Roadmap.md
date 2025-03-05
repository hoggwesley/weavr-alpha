## **Weavr Development Plan: Chronological Implementation Roadmap**


**Last Updated:** February 25, 2025  
**Current Completion:** 24%

---
## Introduction to Weavr AI

Weavr AI is an intelligent knowledge interface that creates a bidirectional connection between personal knowledge systems and cutting-edge AI capabilities. Designed for seamless Obsidian integration through filesystem interactions, the system enables natural language querying and semantic analysis of personal notes while maintaining local data isolation, only connecting externally for secure AI inference via Together API (please note that there is no planned integration of locally run LLM's at this time).

## Core Functionality

The application operates as a self-contained desktop environment that establishes read/write access to user-specified directories, processes queries through an enhanced RAG pipeline, and generates context-aware responses grounded in personal knowledge. Key operational capabilities include manual directory configuration for note storage systems, multi-stage reasoning enhancement processing, and markdown-based content generation with strict filesystem permission validation.

## Technical Architecture

Weavr's refined architecture implements a hybrid Retrieval-Augmented Generation pipeline optimized for Mixtral 8x7B's MoE capabilities, combining semantic search with enhanced reasoning modules. The system processes documents through parallel embedding generation, stores vectors in ChromaDB with automatic refresh cycles, and executes queries using contextually retrieved knowledge enhanced through chained reasoning processes.

## Tools and Libraries

## Core Components:

- **ChromaDB**: Maintains document embeddings with automatic TTL refresh    
- **Together AI API**: Mixtral 8x7B integration with dual-mode operation    
- **Watchdog**: Implements filesystem monitoring with debounced refresh
    
## Frontend:

- **React**: Component-based interface with Redux state management    
- **Electron**: Cross-platform desktop packaging with native FS access    
- **Reasoning Controls**: Custom toggle group for CoT/ToT activation
    
## Backend:

- **Python 3.11**: Async core with type-annotated codebase    
- **FastAPI**: Strictly typed endpoints with OAuth2 security    
- **Reasoning Engine**: Modular processor with validation layer
- **Docker**: Containerization for consistent deployment

## AI Reasoning Modes:

- **Mixtral 8x7B**: Single base model accessed via Together AI with 32K context window    
- **Reasoning Toggles**: Independent enhancement modules that can be activated:    
    - **Chain-of-Thought (CoT)**: Stepwise reasoning for complex problem decomposition        
    - **Tree-of-Thought (ToT)**: Multi-branch exploration of solution pathways        
    - **Consistency Validation**: Verification system to ensure output aligns with source knowledge
        
- **Generation Parameters**: User-adjustable controls via slider interface:    
    - **Temperature**: Controls response creativity and randomness        
    - **Output Length**: Determines response verbosity and detail level
    
## Deployment:

- **Electron Packager**: Platform-specific builds with code signing    
- **Inno Setup**: Windows installer with first-run configuration        


Weavr represents a bridge between your personal knowledge management system and the capabilities of modern AI, allowing you to interact with your notes in ways previously impossible, only connecting to external services for AI inference.

---
## **Phase 1: Core Functionality

#### 1. Reasoning Enhancement Implementation  
**Objective:** Enable granular control over Mixtral 8x7B's reasoning capabilities with parameter tuning  

```yaml
# config.yaml Update
together_ai:
  api_key: "your-api-key"
  mixtral_config:
    base_model: "togethercomputer/Mixtral-8x7B-Instruct-v0.1"
    enhanced_mode: "togethercomputer/Mixtral-8x7B-Instruct-v0.1:enhanced"
  default_params:
    temperature: 0.7
    max_length: 1024
    top_p: 0.9
    repetition_penalty: 1.1
```

**UI Implementation:**  

```jsx
// ReasoningControls.jsx
import RangeSlider from 'react-range-slider-input';
import 'react-range-slider-input/dist/style.css';

export default function ReasoningControls() {
  const [modules, setModules] = useState({
    cot: false,
    tot: false,
    validation: true
  });
  
  const [params, setParams] = useState({
    temperature: 0.7,
    maxLength: 1024
  });

  const handleModuleToggle = (module) => {
    setModules(prev => ({
      ...prev,
      [module]: !prev[module]
    }));
  };

  const generateRequest = async () => {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        modules: Object.keys(modules).filter(m => modules[m]),
        parameters: params,
        prompt: currentPrompt
      })
    });
    
    return response.json();
  };

  return (
    <>
      <button onClick={() => handleModuleToggle('cot')}>Chain-of-Thought</button>
      <button onClick={() => handleModuleToggle('tot')}>Tree-of-Thought</button>
      <button onClick={() => handleModuleToggle('validation')}>Consistency Check</button>
      
      <label>
        Temperature ({params.temperature.toFixed(2)})
        <RangeSlider 
          value={[0.1, params.temperature]} 
          onChange={val => setParams(p => ({...p, temperature: val[1]}))} 
          thumbsDisabled={[true, false]} 
        />
      </label>
      
      <label>
        Response Length ({params.maxLength} tokens)
        <RangeSlider 
          value={[256, params.maxLength]} 
          onChange={val => setParams(p => ({...p, maxLength: val[1]}))} 
          thumbsDisabled={[true, false]} 
        />
      </label>
    </>
  );
}
```

**Backend Implementation:**  

```python
# reasoning_engine.py
class EnhancedProcessor:
    def __init__(self):
        self.modules = {
            'cot': ChainOfThoughtProcessor(),
            'tot': TreeOfThoughtExplorer(),
            'validation': ConsistencyValidator()
        }
    
    async def process_query(self, query, context, active_modules, parameters):
        base_payload = {
            "model": self.config.mixtral_config['base_model'],
            "prompt": query,
            "temperature": parameters['temperature'],
            "max_tokens": parameters['max_length'],
            "top_p": parameters.get('top_p', 0.9),
            "repetition_penalty": parameters.get('repetition_penalty', 1.1)
        }
        
        if any(active_modules.values()):
            base_payload["model"] = self.config.mixtral_config['enhanced_mode']
            
        response = await self.api_call(base_payload)
        
        # Apply active processing modules
        if active_modules['cot']:
            response = self.modules['cot'].process(response, context)
        if active_modules['tot']:
            response = self.modules['tot'].explore(response, context)
        if active_modules['validation']:
            response = self.modules['validation'].validate(response, context)
            
        return response
```

```python
# api_endpoints.py
@app.post("/generate")
async def generate_response(request: GenerateRequest):
    processor = EnhancedProcessor()
    return await processor.process_query(
        request.prompt,
        context=get_context(request.prompt),
        active_modules=request.modules,
        parameters=request.parameters
    )
```

**Key Implementation Details:**  

1. **State Management**  
- Dedicated Redux store for reasoning parameters  
- Local storage persistence for user preferences  
- Debounced API updates for parameter changes  

2. **Validation Rules**  

```python
# validation_rules.py
class ParameterValidator:
    @staticmethod
    def validate_params(params):
        errors = []
        if not (0 <= params['temperature'] <= 1):
            errors.append("Temperature out of range (0-1)")
        if params['max_length'] <= 0:
            errors.append("Invalid max length")
        if errors:
            raise InvalidParametersError(", ".join(errors)
        )
```

3. **Performance Considerations**  
- Web Workers for slider input processing  
- Memoization of common parameter combinations  
- Batch processing of validation rules  

4. **Error Handling**  
- Fallback to base model if enhanced mode fails  
- Automatic parameter clamping for out-of-range values  
- Visual feedback for conflicting module combinations  

**Testing Protocol:**  

```python
# test_reasoning.py
class ReasoningTestCase(unittest.TestCase):
    def test_module_combinations(self):
        test_cases = [
            {'modules': ['cot'], 'expected': True},
            {'modules': ['tot', 'validation'], 'expected': True},
            {'modules': ['cot', 'tot', 'validation'], 'expected': True}
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                result = processor.validate_modules(case['modules'])
                self.assertEqual(result, case['expected'])
```

## **2. Directory Management System**

**Objective:** Implement secure manual directory configuration

```javascript
// Directory Input Component
const DirectorySelector = () => {
  const [paths, setPaths] = useState({
    read: '',
    write: ''
  });
  
  const [errors, setErrors] = useState({});

  const validatePath = async (path, isWrite) => {
    try {
      const res = await fetch('/api/validate-path', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ path, isWrite })
      });
      
      const { valid, error } = await res.json();
      if (!valid) setErrors(prev => ({ ...prev, [isWrite ? 'write' : 'read']: error }));
      return valid;
    } catch (err) {
      setErrors(prev => ({ ...prev, system: 'Validation service unavailable' }));
      return false;
    }
  };

  const handlePathChange = async (type, value) => {
    const isValid = await validatePath(value, type === 'write');
    if (isValid) {
      setPaths(prev => ({ ...prev, [type]: value }));
      setErrors(prev => ({ ...prev, [type]: undefined }));
    }
  };

  return (
    <div className="directory-manager">
      <div className="path-input">
        <label>Knowledge Base Directory:</label>
        <input
          type="text"
          value={paths.read}
          onChange={(e) => handlePathChange('read', e.target.value)}
          placeholder="/path/to/obsidian/vault"
          className={errors.read ? 'invalid' : ''}
        />
        {errors.read && <div className="error-text">{errors.read}</div>}
      </div>
      
      <div className="path-input">
        <label>Output Directory:</label>
        <input
          type="text"
          value={paths.write}
          onChange={(e) => handlePathChange('write', e.target.value)}
          placeholder="/path/to/output/folder"
          className={errors.write ? 'invalid' : ''}
        />
        {errors.write && <div className="error-text">{errors.write}</div>}
      </div>
      
      {errors.system && <div className="system-error">{errors.system}</div>}
    </div>
  );
};
```

**File Handling:**

```python
import os
import re

class ValidationError(Exception):
    pass

class SecurityError(Exception):
    pass

def validate_directory(path: str, require_write: bool = True) -> bool:
    """Validate directory permissions and structure"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not os.path.isdir(path):
            raise NotADirectoryError(f"Not a directory: {path}")

        if require_write and not os.access(path, os.W_OK):
            raise PermissionError(f"Write access denied: {path}")

        # Prevent symlink attacks
        if os.path.islink(path):
            raise SecurityError("Symlinks not allowed for security reasons")

        return True
    except Exception as e:
        raise ValidationError(str(e)) from e

def write_markdown(content: str, directory: str, filename: str) -> str:
    """Safely write processed content to markdown file"""
    validated = validate_directory(directory)
    safe_filename = re.sub(r'[^\w\-_]', '', filename)[:255]

    full_path = os.path.join(directory, f"{safe_filename}.md")
    with open(full_path, 'w', encoding='utf-8', errors='xmlcharrefreplace') as f:
        f.write(content)

    return os.path.normpath(full_path)
```

**Key Enhancements:**

1. Added real-time path validation API integration    
2. Implemented security checks against symlink attacks    
3. Added error state visualization in UI components    
4. Included filename sanitization and path normalization    
5. Added comprehensive permission validation    
6. Implemented cross-platform path handling
    

---

### **Phase 2: Installation & Configuration**  
#### **1. Installer Development**  
**Objective:** Create cross-platform installer with secure directory configuration  

```iss
; Inno Setup Script
[Setup]
AppName=Weavr
AppVersion=1.0
DefaultDirName={userdocs}\Weavr
OutputDir=userdocs

[Code]
procedure InitializeWizard;
begin
  WizardForm.DirEdit.Text := ExpandConstant('{userdocs}');
end;
```

**First-Run Setup Sequence:**  
1. Together AI API key configuration  
2. Secure directory selection with validation  
3. Reasoning enhancement preferences  
4. Parameter slider default configuration  

---

#### **2. File System Integration Layer**  
**Objective:** Implement secure directory monitoring with validation  

```python
# File Watcher Service with Security Validation
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class SecurityValidator:
    @staticmethod
    def validate_path(path):
        if os.path.islink(path):
            return False  # Prevent symlink attacks
        return os.path.exists(path) and os.access(path, os.R_OK)

class DirectoryHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.validator = SecurityValidator()
    
    def on_modified(self, event):
        if not event.is_directory and self.validator.validate_path(event.src_path):
            self.processor.process_file(event.src_path)

observer = Observer()
observer.schedule(DirectoryHandler(processor), path=validated_read_path, recursive=True)
observer.start()
```

The key changes include:
1. Removing model preference setup in favor of reasoning enhancement preferences  
2. Adding parameter slider configuration to the first-run sequence  
3. Implementing security validation for file paths  
4. Replacing direct Obsidian references with more generic "File System Integration"  
5. Adding symlink attack prevention in the file watcher service  

---

### **Phase 3: Application Packaging**  
#### **1. Electron Packaging**  

```bash
# Packaging Commands
npm install electron --save-dev
npx electron-packager . Weavr --platform=darwin,win32 --arch=x64 --out=dist
```

#### **2. Docker Alternative**  

```yaml
# docker-compose.yml
version: '3.8'
services:
  weavr-core:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - /user/specified/path:/data
    environment:
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
```

---

### **Phase 4: Extended File Support**  
**Future Implementation:**  

```javascript
// File Format Selector (Phase 4)
const FormatSelector = () => (
  <select>
    <option>Markdown</option>
    <option>Plain Text</option>
    <option>JSON</option>
  </select>
);
```

---

### **Revised Development Timeline**  

| Component               | Time Estimate | Difficulty |  
|-------------------------|---------------|------------|  
| Reasoning Enhancement   | 8 hours       | Medium     |  
| Directory System        | 8 hours       | Medium     |  
| Installer Wizard        | 1 day         | Medium     |  
| File Monitoring         | 6 hours       | Hard       |  
| Electron Packaging      | 2 days        | Medium     |  
| **Total**               | **10 days**   |            |  

---

### **System Architecture**  

```
[User Interface] ↔ [Weavr Core] ↔ [Together AI API (Mixtral 8x7B)]  
       │                 │
[Read Directory] ↔ [ChromaDB]  
       │                 │
[Write Directory] ← [Reasoning Engine]  
       │
Markdown Files
```

---

This revised roadmap implements the Mixtral 8x7B model with reasoning enhancements while maintaining the manual directory configuration approach. The architecture now includes a dedicated reasoning engine component that processes retrieved knowledge through optional enhancement modules (CoT, ToT, and Validation).

**Citations:**  
[1] [Weavr Pivot Strategy](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/52646530/755291f3-6889-4ac1-901d-ff2d2b2630f7/Weavr-Pivot-Strategy.md)  
[2] [Weavr AI v1.0 Development Roadmap](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_427ab709-549f-4e83-b2a8-0c1d5489cf70/70c8ddea-17b8-4a00-a1b8-adb58920b55f/Weavr-AI-v1.0-Development-Roadmap.md)  
[3] [Weavr AI v1.0 Current Project Status](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_427ab709-549f-4e83-b2a8-0c1d5489cf70/4402477b-392c-42c6-a37e-2cef1989b0fd/Weavr-AI-v1.0-Current-Project-Status.md)  
[4] [Custom Build Hardware System Specifications](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_427ab709-549f-4e83-b2a8-0c1d5489cf70/1b58f1f8-5cf2-4059-94c4-954b42d60dd8/Custom-Build-Hardware-System-Specifications.md)  
