import asyncio
import json
import os
from pathlib import Path
from typing import Any, Hashable

import pandas as pd
from pydantic import Field, model_validator

from app.config import config
from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT, BaseSandboxClient
from app.tool.base import BaseTool


class DataVisualization(BaseTool):
    name: str = "data_visualization"
    description: str = """Visualize statistical chart or Add insights in chart with JSON info from visualization_preparation tool. You can do steps as follows:
1. Visualize statistical chart
2. Choose insights into chart based on step 1 (Optional)
Outputs:
1. Charts (png/html)
2. Charts Insights (.md)(Optional)"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "json_path": {
                "type": "string",
                "description": """file path of json info with ".json" in the end""",
            },
            "output_type": {
                "description": "Rendering format (html=interactive)",
                "type": "string",
                "default": "html",
                "enum": ["png", "html"],
            },
            "tool_type": {
                "description": "visualize chart or add insights",
                "type": "string",
                "default": "visualization",
                "enum": ["visualization", "insight"],
            },
            "language": {
                "description": "english(en) / chinese(zh)",
                "type": "string",
                "default": "en",
                "enum": ["zh", "en"],
            },
        },
        "required": ["code"],
    }
    llm: LLM = Field(default_factory=LLM, description="Language model instance")

    @model_validator(mode="after")
    def initialize_llm(self):
        """Initialize llm with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        return self

    def get_file_path(
        self,
        json_info: list[dict[str, str]],
        path_str: str,
        directory: str = None,
    ) -> list[str]:
        res = []
        for item in json_info:
            if os.path.exists(item[path_str]):
                res.append(item[path_str])
            elif os.path.exists(
                os.path.join(
                    f"{directory or config.workspace_root}",
                    item[path_str],
                )
            ):
                res.append(
                    os.path.join(
                        f"{directory or config.workspace_root}",
                        item[path_str],
                    )
                )
            else:
                if config.sandbox.use_sandbox and config.sandbox.shared_workspace:
                    if item[path_str].startswith(config.sandbox.work_dir):
                        relative = Path(item[path_str]).relative_to(
                            Path(config.sandbox.work_dir)
                        )
                        new_path = os.path.join(
                            f"{directory or config.workspace_root}",
                            str(relative),
                        )
                        if os.path.exists(new_path):
                            res.append(new_path)
                            continue
                raise Exception(f"No such file or directory: {item[path_str]}")
        return res

    def success_output_template(self, result: list[dict[str, str]]) -> str:
        content = ""
        if len(result) == 0:
            return "Is EMPTY!"
        for item in result:
            content += f"""## {item['title']}\nChart saved in: {item['chart_path']}"""
            if "insight_path" in item and item["insight_path"] and "insight_md" in item:
                content += "\n" + item["insight_md"]
            else:
                content += "\n"
        return f"Chart Generated Successful!\n{content}"

    async def data_visualization(
        self, json_info: list[dict[str, str]], output_type: str, language: str
    ) -> str:
        data_list = []
        csv_file_path = self.get_file_path(json_info, "csvFilePath")
        for index, item in enumerate(json_info):
            df = pd.read_csv(csv_file_path[index], encoding="utf-8")
            df = df.astype(object)
            df = df.where(pd.notnull(df), None)
            data_dict_list = df.to_json(orient="records", force_ascii=False)

            data_list.append(
                {
                    "file_name": os.path.basename(csv_file_path[index]).replace(
                        ".csv", ""
                    ),
                    "dict_data": data_dict_list,
                    "chartTitle": item["chartTitle"],
                }
            )

        results = []
        if config.sandbox.use_sandbox:
            try:
                await self._ensure_sandbox_initialized()
                for item in data_list:
                    result = await self.invoke_vmind(
                        dict_data=item["dict_data"],
                        chart_description=item["chartTitle"],
                        file_name=item["file_name"],
                        output_type=output_type,
                        task_type="visualization",
                        language=language,
                    )
                    results.append(result)
            finally:
                await self.sandbox_client.cleanup()
        else:
            tasks = [
                self.invoke_vmind(
                    dict_data=item["dict_data"],
                    chart_description=item["chartTitle"],
                    file_name=item["file_name"],
                    output_type=output_type,
                    task_type="visualization",
                    language=language,
                )
                for item in data_list
            ]
            results = await asyncio.gather(*tasks)
        error_list = []
        success_list = []
        for index, result in enumerate(results):
            csv_path = csv_file_path[index]
            if "error" in result and "chart_path" not in result:
                error_list.append(f"Error in {csv_path}: {result['error']}")
            else:
                success_list.append(
                    {
                        **result,
                        "title": json_info[index]["chartTitle"],
                    }
                )
        if len(error_list) > 0:
            return {
                "observation": f"# Error chart generated{'\n'.join(error_list)}\n{self.success_output_template(success_list)}",
                "success": False,
            }
        else:
            return {"observation": f"{self.success_output_template(success_list)}"}

    async def add_insighs(
        self, json_info: list[dict[str, str]], output_type: str
    ) -> str:
        data_list = []
        chart_file_path = self.get_file_path(
            json_info,
            "chartPath",
            os.path.join(config.workspace_root_or_sandbox_work_dir, "visualization"),
        )
        for index, item in enumerate(json_info):
            if "insights_id" in item:
                data_list.append(
                    {
                        "file_name": os.path.basename(chart_file_path[index]).replace(
                            f".{output_type}", ""
                        ),
                        "insights_id": item["insights_id"],
                    }
                )
        results = []
        if config.sandbox.use_sandbox:
            try:
                await self._ensure_sandbox_initialized()
                for item in data_list:
                    result = await self.invoke_vmind(
                        insights_id=item["insights_id"],
                        file_name=item["file_name"],
                        output_type=output_type,
                        task_type="insight",
                    )
                    results.append(result)
            finally:
                await self.sandbox_client.cleanup()
        else:
            tasks = [
                self.invoke_vmind(
                    insights_id=item["insights_id"],
                    file_name=item["file_name"],
                    output_type=output_type,
                    task_type="insight",
                )
                for item in data_list
            ]
            results = await asyncio.gather(*tasks)
        error_list = []
        success_list = []
        for index, result in enumerate(results):
            chart_path = chart_file_path[index]
            if "error" in result and "chart_path" not in result:
                error_list.append(f"Error in {chart_path}: {result['error']}")
            else:
                success_list.append(chart_path)
        success_template = (
            f"# Charts Update with Insights\n{','.join(success_list)}"
            if len(success_list) > 0
            else ""
        )
        if len(error_list) > 0:
            return {
                "observation": f"# Error in chart insights:{'\n'.join(error_list)}\n{success_template}",
                "success": False,
            }
        else:
            return {"observation": f"{success_template}"}

    async def execute(
        self,
        json_path: str,
        output_type: str | None = "html",
        tool_type: str | None = "visualization",
        language: str | None = "en",
    ) -> str:
        try:
            logger.info(f"📈 data_visualization with {json_path} in: {tool_type} ")

            json_info = await self.json_load(json_path)
            if isinstance(json_info, dict):
                json_info = [json_info]

            if tool_type == "visualization":
                return await self.data_visualization(json_info, output_type, language)
            else:
                return await self.add_insighs(json_info, output_type)
        except Exception as e:
            return {
                "observation": f"Error: {e}",
                "success": False,
            }

    sandbox_client: BaseSandboxClient = Field(None)

    async def _ensure_sandbox_initialized(self):
        """Ensure sandbox is initialized."""
        if config.sandbox.use_sandbox and self.sandbox_client == None:
            self.sandbox_client = SANDBOX_CLIENT

        if self.sandbox_client:
            await self.sandbox_client.create(config.sandbox)

    async def json_load(self, json_path):
        if config.sandbox.use_sandbox:
            try:
                await self._ensure_sandbox_initialized()
                content = await self.sandbox_client.read_file(json_path)
                return json.loads(content)
            finally:
                await self.sandbox_client.cleanup()
        else:
            with open(json_path, "r", encoding="utf-8") as file:
                return json.load(file)

    async def invoke_vmind(
        self,
        file_name: str,
        output_type: str,
        task_type: str,
        insights_id: list[str] = None,
        dict_data: list[dict[Hashable, Any]] = None,
        chart_description: str = None,
        language: str = "en",
    ):
        llm_config = {
            "base_url": self.llm.base_url,
            "model": self.llm.model,
            "api_key": self.llm.api_key,
        }
        vmind_params = {
            "llm_config": llm_config,
            "user_prompt": chart_description,
            "dataset": dict_data,
            "file_name": file_name,
            "output_type": output_type,
            "insights_id": insights_id,
            "task_type": task_type,
            "directory": str(config.workspace_root_or_sandbox_work_dir),
            "language": language,
        }

        input_json = json.dumps(vmind_params, ensure_ascii=False).encode("utf-8")

        if config.sandbox.use_sandbox:
            timeout = config.sandbox.timeout
            if config.sandbox.shared_workspace:
                # Write code.py to workspace
                with open(config.workspace_root / "vmind_input.json", "wb") as f:
                    f.write(input_json)
            else:
                await self.sandbox_client.write_file(
                    "/workspace/vmind_input.json", input_json
                )
            result = await self.sandbox_client.run_command(
                "cd /chart_visualization/ && npx ts-node ./src/chartVisualize.ts < /workspace/vmind_input.json",
                timeout,
            )
            try:
                return json.loads(result)
            except Exception as e:
                return {"error": f"Subprocess Error: {result}"}
        else:
            # build async sub process
            process = await asyncio.create_subprocess_exec(
                "npx",
                "ts-node",
                "src/chartVisualize.ts",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(__file__),
            )

            try:
                stdout, stderr = await process.communicate(input_json)
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")
                if process.returncode == 0:
                    return json.loads(stdout_str)
                else:
                    return {"error": f"Node.js Error: {stderr_str}"}
            except Exception as e:
                return {"error": f"Subprocess Error: {str(e)}"}
