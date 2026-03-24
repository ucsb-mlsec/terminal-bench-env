import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


@dataclass
class BashCommand:
    """Represents a command to be executed by the agent."""
    command: str
    timeout_sec: int = 120


@dataclass
class BashParseResult:
    """Result of parsing a model response."""
    commands: List[BashCommand]
    is_task_complete: bool
    error: str
    warning: str


class BashAgent(BaseAgent):
    """Minimal ReAct-style agent for Harbor framework.

    Compatible with Qwen2.5-Coder, TermiGen, and other models that support
    the <tool_call> XML format for bash command execution.
    """

    @staticmethod
    def name() -> str:
        return "bash-agent"

    def version(self) -> str:
        return "1.0.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        model_endpoint: str = "http://172.17.0.1:8001/v1",
        max_episodes: int = 1000,
        temperature: float = 0.6,
        **kwargs
    ):
        """
        Initialize BashAgent for Harbor framework.

        Args:
            logs_dir: Directory for logging
            model_name: Name of the model (provider/model format)
            model_endpoint: URL endpoint for the model API (OpenAI-compatible)
            max_episodes: Maximum number of conversation episodes
            temperature: Temperature for model generation
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.model_endpoint = os.getenv("MODEL_ENDPOINT", model_endpoint)
        self._model_name_for_api = os.getenv("MODEL_NAME", model_name or "")
        self.temperature = temperature
        self._max_episodes = max_episodes

        # Conversation state
        self._conversation_history: List[Dict[str, str]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.model_endpoint
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent - no special setup needed."""
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Run the agent in the environment.

        Args:
            instruction: The task instruction.
            environment: The environment to execute commands in.
            context: The context to populate with execution results.
        """
        # Reset conversation state
        self._conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are an expert technical assistant with access to bash tools. "
                    "You can execute bash commands to help solve complex technical problems. "
                    "When you need to run commands, use the bash tool with the following format:\n\n"
                    "<tool_call>\n"
                    "{\"name\": \"bash\", \"arguments\": {\"command\": \"your_command_here\"}}\n"
                    "</tool_call>\n\n"
                    "Always think through problems step by step, analyze the situation, "
                    "and then execute appropriate commands to solve the task. "
                    "You can run multiple commands in sequence and analyze their outputs "
                    "to make informed decisions."
                )
            },
        ]
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        try:
            await self._run_conversation_loop(instruction, environment)
        finally:
            # Always populate context with token counts
            context.n_input_tokens = self._total_input_tokens
            context.n_output_tokens = self._total_output_tokens

    async def _run_conversation_loop(
        self,
        instruction: str,
        environment: BaseEnvironment,
    ) -> None:
        """Run the main conversation loop with the model."""
        last_output = ""

        for episode in range(self._max_episodes):
            # Setup logging for this episode
            episode_dir = self.logs_dir / f"episode-{episode}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Get response from model
            if episode == 0:
                response = self._query_model(instruction, episode_dir, is_observation=False)
            else:
                observation = self._limit_output_length(last_output)
                self.logger.debug(f"Observation: {observation[:500]}...")
                response = self._query_model(observation, episode_dir, is_observation=True)

            self._conversation_history.append({"role": "assistant", "content": response})

            # Parse the response
            parse_result = self._parse_response(response)

            # Handle parsing errors
            if parse_result.error:
                self.logger.warning(f"Parsing error in episode {episode}: {parse_result.error}")

            # Check if agent proposed any tool calls
            if not parse_result.commands:
                self.logger.info(f"No tool calls proposed by agent (episode {episode}), stopping")
                break

            # Execute commands and collect output
            last_output = await self._execute_commands(parse_result.commands, environment)

            # Check for task completion
            if parse_result.is_task_complete:
                self.logger.info(f"Task marked as complete at episode {episode}")
                break

    async def _execute_commands(
        self,
        commands: List[BashCommand],
        environment: BaseEnvironment,
    ) -> str:
        """Execute a list of commands in the environment and return combined output."""
        outputs = []

        for command in commands:
            try:
                self.logger.info(f"Executing command: {command.command}")
                result = await environment.exec(
                    command=command.command,
                    timeout_sec=command.timeout_sec,
                )

                output_parts = []
                if result.stdout:
                    output_parts.append(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    # Filter out non-interactive shell warnings
                    stderr_lines = [
                        line for line in result.stderr.splitlines()
                        if "cannot set terminal process group" not in line
                        and "no job control in this shell" not in line
                    ]
                    if stderr_lines:
                        output_parts.append(f"STDERR:\n{chr(10).join(stderr_lines)}")
                output_parts.append(f"EXIT CODE: {result.return_code}")

                outputs.append("\n".join(output_parts))

            except TimeoutError:
                self.logger.warning(f"Command timed out: {command.command}")
                outputs.append(f"Command timed out after {command.timeout_sec}s: {command.command}")
            except Exception as e:
                self.logger.error(f"Error executing command: {e}")
                outputs.append(f"Error executing command: {e}")

        return "\n\n---\n\n".join(outputs)

    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """Limit output to specified byte length, keeping first and last portions."""
        if len(output.encode("utf-8")) <= max_bytes:
            return output

        portion_size = max_bytes // 2
        output_bytes = output.encode("utf-8")

        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )

        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    def _query_model(
        self,
        prompt: str,
        episode_dir: Path,
        is_observation: bool = False,
    ) -> str:
        """Query the model with the given prompt."""
        # Save prompt
        prompt_path = episode_dir / "prompt.txt"
        prompt_path.write_text(prompt)

        # Prepare messages
        messages = list(self._conversation_history)

        if is_observation:
            messages.append({"role": "tool", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        # Update conversation history
        self._conversation_history = messages

        try:
            chat_response = self.client.chat.completions.create(
                model=self._model_name_for_api,
                messages=messages,
                max_tokens=5000,
                temperature=self.temperature,
            )

            # Update token counts
            if hasattr(chat_response, 'usage') and chat_response.usage:
                self._total_input_tokens += chat_response.usage.prompt_tokens
                self._total_output_tokens += chat_response.usage.completion_tokens

            # Get response content
            message = chat_response.choices[0].message
            response_content = message.content

            # vLLM may put content in reasoning_content
            if response_content is None and hasattr(message, 'reasoning_content'):
                response_content = message.reasoning_content

            if response_content is None:
                raise Exception("Model returned None response content")

            # Save response
            response_path = episode_dir / "response.txt"
            response_path.write_text(response_content)

            return response_content

        except Exception as e:
            raise Exception(f"Failed to call model: {e}")

    def _parse_response(self, response: str) -> BashParseResult:
        """Parse the model response and extract commands and completion status."""
        try:
            tool_calls = self._extract_tool_calls(response)

            commands = []
            for tool_call in tool_calls:
                if tool_call.get("name") == "bash":
                    command = tool_call.get("arguments", {}).get("command", "")
                    if command:
                        commands.append(BashCommand(command=command, timeout_sec=120))

            is_task_complete = self._check_task_completion(response) or len(commands) == 0

            return BashParseResult(
                commands=commands,
                is_task_complete=is_task_complete,
                error="",
                warning=""
            )

        except Exception as e:
            return BashParseResult(
                commands=[],
                is_task_complete=False,
                error=f"Error parsing response: {e}",
                warning=""
            )

    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from the response content."""
        tool_calls = []

        if not content:
            return tool_calls

        # Find all tool_call blocks
        patterns = [
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            r'<tool_call>\s*(.*?)\s*</tool_call>',
        ]

        matches = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                break

        for match in matches:
            match = match.strip()
            if not match:
                continue

            tool_call = self._parse_single_tool_call(match)
            if tool_call:
                tool_calls.append(tool_call)

        return tool_calls

    def _parse_single_tool_call(self, match: str) -> Optional[Dict[str, Any]]:
        """Parse a single tool call with multiple fallback strategies."""
        # Strategy 1: Direct JSON parsing
        try:
            tool_call = json.loads(match)
            if self._validate_tool_call(tool_call):
                return tool_call
        except json.JSONDecodeError:
            pass

        # Strategy 2: Fix common JSON issues
        try:
            fixed = re.sub(r',\s*}', '}', match)
            fixed = re.sub(r',\s*]', ']', fixed)
            tool_call = json.loads(fixed)
            if self._validate_tool_call(tool_call):
                return tool_call
        except json.JSONDecodeError:
            pass

        # Strategy 3: Regex extraction
        try:
            name_match = re.search(r'"name":\s*"([^"]*)"', match)
            command_match = re.search(r'"command":\s*"([^"]*(?:\\.[^"]*)*)"', match)

            if name_match and command_match:
                return {
                    "name": name_match.group(1),
                    "arguments": {"command": command_match.group(1).replace('\\"', '"')}
                }
        except Exception:
            pass

        return None

    def _validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """Validate tool call structure."""
        return (
            isinstance(tool_call, dict)
            and "name" in tool_call
            and "arguments" in tool_call
            and isinstance(tool_call["arguments"], dict)
            and "command" in tool_call["arguments"]
        )

    def _check_task_completion(self, content: str) -> bool:
        """Check if the task is marked as complete."""
        indicators = [
            "task_complete: true",
            "task_complete:true",
            "task complete: true",
            "task is complete",
            "task completed",
            "finished the task",
            "task finished"
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in indicators)
