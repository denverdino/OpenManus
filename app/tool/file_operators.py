"""File operation interfaces and implementations for local and sandbox environments."""

import asyncio
from pathlib import Path
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

from app.config import config, SandboxSettings
from app.exceptions import ToolError
from app.sandbox.client import SANDBOX_CLIENT

PathLike = Union[str, Path]


@runtime_checkable
class FileOperator(Protocol):
    """Interface for file operations in different environments."""

    async def read_file(self, path: PathLike) -> str:
        """Read content from a file."""
        ...

    async def write_file(self, path: PathLike, content: str) -> None:
        """Write content to a file."""
        ...

    async def is_directory(self, path: PathLike) -> bool:
        """Check if path points to a directory."""
        ...

    async def exists(self, path: PathLike) -> bool:
        """Check if path exists."""
        ...

    def normalize_path(self, path: PathLike) -> str:
        """Normalize path."""
        ...

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a shell command and return (return_code, stdout, stderr)."""
        ...


class LocalFileOperator(FileOperator):
    """File operations implementation for local filesystem."""

    encoding: str = "utf-8"

    async def read_file(self, path: PathLike) -> str:
        """Read content from a local file."""
        try:
            return Path(path).read_text(encoding=self.encoding)
        except Exception as e:
            raise ToolError(f"Failed to read {path}: {str(e)}") from None

    async def write_file(self, path: PathLike, content: str) -> None:
        """Write content to a local file."""
        try:
            Path(path).write_text(content, encoding=self.encoding)
        except Exception as e:
            raise ToolError(f"Failed to write to {path}: {str(e)}") from None

    async def is_directory(self, path: PathLike) -> bool:
        """Check if path points to a directory."""
        return Path(path).is_dir()

    async def exists(self, path: PathLike) -> bool:
        """Check if path exists."""
        return Path(path).exists()

    def normalize_path(self, path: PathLike) -> str:
        """Normalize path to local filesystem."""
        return str(path)

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a shell command locally."""
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            return (
                process.returncode or 0,
                stdout.decode(),
                stderr.decode(),
            )
        except asyncio.TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            raise TimeoutError(
                f"Command '{cmd}' timed out after {timeout} seconds"
            ) from exc


class SandboxFileOperator(FileOperator):
    """File operations implementation for sandbox environment."""

    def __init__(self):
        self.sandbox_client = SANDBOX_CLIENT

    def normalize_path(self, path: PathLike) -> str:
        """Normalize path to sandbox workdir"""
        # If path is relative to workspace root, convert it to relative to sandbox workdir
        if Path(path).is_relative_to(config.workspace_root):
            # Convert to relative path from sandbox workdir
            return str(
                Path(config.sandbox.work_dir).joinpath(
                    Path(path).relative_to(config.workspace_root)
                )
            )
        return path

    async def _ensure_sandbox_initialized(self):
        """Ensure sandbox is initialized."""
        if not self.sandbox_client.sandbox:
            await self.sandbox_client.create(config=config.sandbox)

    async def read_file(self, path: PathLike) -> str:
        """Read content from a file in sandbox."""
        try:
            normalized_path = self.normalize_path(path)
            await self._ensure_sandbox_initialized()
            return await self.sandbox_client.read_file(normalized_path)
        except Exception as e:
            raise ToolError(f"Failed to read {path} in sandbox: {str(e)}") from None
        finally:
            await self.sandbox_client.cleanup()

    async def write_file(self, path: PathLike, content: str) -> None:
        """Write content to a file in sandbox."""
        try:
            normalized_path = self.normalize_path(path)
            await self._ensure_sandbox_initialized()
            await self.sandbox_client.write_file(normalized_path, content)
        except Exception as e:
            raise ToolError(f"Failed to write to {path} in sandbox: {str(e)}") from None
        finally:
            await self.sandbox_client.cleanup()

    async def is_directory(self, path: PathLike) -> bool:
        """Check if path points to a directory in sandbox."""
        try:
            normalized_path = self.normalize_path(path)
            await self._ensure_sandbox_initialized()
            result = await self.sandbox_client.run_command(
                f"test -d {normalized_path} && echo 'true' || echo 'false'"
            )
            return result.strip() == "true"
        finally:
            await self.sandbox_client.cleanup()

    async def exists(self, path: PathLike) -> bool:
        """Check if path exists in sandbox."""
        try:
            normalized_path = self.normalize_path(path)
            await self._ensure_sandbox_initialized()
            result = await self.sandbox_client.run_command(
                f"test -e {normalized_path} && echo 'true' || echo 'false'"
            )
            return result.strip() == "true"
        finally:
            await self.sandbox_client.cleanup()

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a command in sandbox environment."""
        try:
            await self._ensure_sandbox_initialized()
            stdout = await self.sandbox_client.run_command(
                cmd, timeout=int(timeout) if timeout else None
            )
            return (
                0,  # Always return 0 since we don't have explicit return code from sandbox
                stdout,
                "",  # No stderr capture in the current sandbox implementation
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Command '{cmd}' timed out after {timeout} seconds in sandbox"
            ) from exc
        except Exception as exc:
            return 1, "", f"Error executing command in sandbox: {str(exc)}"
        finally:
            await self.sandbox_client.cleanup()
