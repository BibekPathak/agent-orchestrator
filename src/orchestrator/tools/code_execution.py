from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from ..core.tool import Tool


class PythonExecutionTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="python_execute",
            description="Execute Python code and return the output. Use this for calculations, data processing, and running scripts.",
        )

    async def execute(self, code: str) -> str:
        """Execute Python code and capture stdout, stderr, and return value."""
        # Security: Only allow safe operations
        # Block dangerous imports and operations
        forbidden_patterns = [
            'import os', 'import subprocess', 'import sys',
            '__import__', 'eval(', 'exec(', 'compile(',
            'open(', 'file(', 'input(', 'exit(', 'quit(',
            'globals()', 'locals()', 'vars()', 'getattr(',
            'setattr(', 'delattr(', 'hasattr(',
            'shutil', 'socket', 'urllib', 'requests',
        ]
        
        # Check for forbidden patterns
        for pattern in forbidden_patterns:
            if pattern in code:
                return f"Error: Forbidden pattern detected: {pattern}"
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Execute the code with restricted globals
            # Only allow basic builtins and safe modules
            safe_globals = {
                '__builtins__': {
                    'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                    'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'range': range, 'enumerate': enumerate, 'zip': zip, 'sorted': sorted,
                    'reversed': reversed, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
                    'round': round, 'pow': pow, 'divmod': divmod, 'all': all, 'any': any,
                    'isinstance': isinstance, 'issubclass': issubclass, 'hash': hash,
                    'type': type, 'repr': repr, 'format': format, 'input': input,
                },
                'math': __import__('math'),
                'random': __import__('random'),
                'datetime': __import__('datetime'),
                'json': __import__('json'),
            }
            
            safe_locals = {}
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, safe_globals, safe_locals)
            
            stdout_result = stdout_capture.getvalue()
            stderr_result = stderr_capture.getvalue()
            
            # Format the output
            output_parts = []
            if stdout_result.strip():
                output_parts.append("Output:\n" + stdout_result.rstrip())
            if stderr_result.strip():
                output_parts.append("Errors:\n" + stderr_result.rstrip())
            
            if not output_parts:
                output_parts.append("Code executed successfully (no output)")
            
            return "\n\n".join(output_parts)
            
        except Exception as e:
            return f"Error executing Python code:\n{traceback.format_exc()}"