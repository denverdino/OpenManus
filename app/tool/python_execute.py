import asyncio
import atexit
import multiprocessing
import sys
from io import StringIO
from typing import Any, Dict

from pydantic import Field

from app.config import config
from app.sandbox.client import SANDBOX_CLIENT, BaseSandboxClient
from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = (
        "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results. If the code depends on any non-builtin modules, install those packages before importing them at the top of the code. for example: \nimport subprocess\nsubprocess.call(['pip', 'install', 'numpy'])\nimport numpy "
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }
    sandbox_client: BaseSandboxClient = Field(None)

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def _ensure_sandbox_initialized(self):
        """Ensure sandbox is initialized."""
        if config.sandbox.use_sandbox and self.sandbox_client == None:
            self.sandbox_client = SANDBOX_CLIENT

        if self.sandbox_client:
            await self.sandbox_client.create(config.sandbox)

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """

        print(code)

        if config.sandbox.use_sandbox:
            timeout = config.sandbox.timeout
            try:
                await self._ensure_sandbox_initialized()
                if config.sandbox.shared_workspace:
                    # Write code.py to workspace
                    with open(
                        config.workspace_root / "code.py", "w", encoding="utf-8"
                    ) as f:
                        f.write(code)
                else:
                    await self.sandbox_client.write_file("/workspace/code.py", code)
                result = await self.sandbox_client.run_command(
                    "python3 code.py", timeout
                )
                return {
                    "observation": result,
                    "success": True,
                }
            finally:
                await self.sandbox_client.cleanup()

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            return dict(result)
