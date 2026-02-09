"""
Qwen Agent implementation for terminal task automation.

This module provides a BashAgent that uses Qwen models to execute bash commands
in a terminal session through multi-turn conversations. The agent parses tool calls
from model responses and executes them sequentially.

Note: This agent is primarily designed for Qwen models. If you want to use it with
other models, you need to verify whether the model supports the "tool" role for
messages. If the model doesn't support the "tool" role, you may need to modify
line 411 in the `_call_qwen_model` method to use "user" role or another supported
role instead of "tool" when sending observations (terminal output).
"""
import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
import time

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.utils.logger import logger


@dataclass
class QwenCommand:
    """Represents a command to be executed by the Qwen agent.
    
    Attributes:
        command: The bash command string to execute
        duration_sec: Expected duration in seconds for command execution (used for timeout)
    """
    command: str
    duration_sec: float = 10.0


@dataclass
class QwenParseResult:
    """Result of parsing a Qwen model response.
    
    Attributes:
        commands: List of extracted commands to execute
        is_task_complete: Whether the agent indicates the task is finished
        error: Error message if parsing failed
        warning: Warning message for non-fatal parsing issues
    """
    commands: List[QwenCommand]
    is_task_complete: bool
    error: str
    warning: str


class BashAgent(BaseAgent):
    """
    Agent that uses Qwen models to execute bash commands in terminal sessions.
    
    The agent maintains a conversation history with the model, sending terminal
    output as observations and receiving tool calls in response. It supports
    multi-turn interactions until the task is complete or max episodes reached.
    """
    @staticmethod
    def name() -> str:
        return "bash-agent"

    def __init__(
        self, 
        model_endpoint: str, 
        model_name: str,
        max_episodes: int = 1000,
        temperature: float = 0.6,
        max_tokens: int = 5000,
        command_duration_sec: float = 10.0,
        **kwargs
    ):
        """
        Initialize Bash Agent.
        
        Args:
            model_endpoint: URL endpoint for the Qwen model API
            model_name: Name of the model to use
            max_episodes: Maximum number of conversation episodes (default: unlimited)
            temperature: Temperature for model generation
            max_tokens: Maximum number of tokens to generate (default: 5000)
            command_duration_sec: Default duration in seconds for command execution (default: 10.0)
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(**kwargs)
        # Allow environment variables to override constructor parameters
        self.model_endpoint = os.getenv("MODEL_ENDPOINT", model_endpoint)
        self.model_name = os.getenv("MODEL_NAME", model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.command_duration_sec = command_duration_sec
        self._logger = logger.getChild(__name__)
        self._max_episodes = max_episodes
        
        # Conversation state - maintained across episodes
        self._conversation_history: List[Dict[str, str]] = []
        self._timestamped_markers: List[Tuple[float, str]] = []  # For asciinema debugging
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
        # Initialize OpenAI client (api_key="EMPTY" is a placeholder for vLLM-compatible APIs)
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.model_endpoint
        )

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """
        Perform a task using Qwen agent with multi-turn conversation support.
        
        Args:
            instruction: The task instruction to execute
            session: Active tmux session for command execution
            logging_dir: Optional directory for episode-level logging
            
        Returns:
            AgentResult with token usage, failure mode, and debugging markers
        """
        # Reset conversation state for new task
        self._conversation_history = [
            {"role": "system", "content": "You are an expert technical assistant with access to bash tools. You can execute bash commands to help solve complex technical problems. When you need to run commands, use the bash tool with the following format:\n\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"your_command_here\"}}\n</tool_call>\n\nAlways think through problems step by step, analyze the situation, and then execute appropriate commands to solve the task. You can run one command at a time and analyze its output to make informed decisions."},
        ]
        self._timestamped_markers = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self.has_tool_call = True  # Track if agent is proposing tool calls
        
        try:
            # Start the conversation loop
            self._run_conversation_loop(instruction, session, logging_dir)
            
            return AgentResult(
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                failure_mode=FailureMode.NONE,
                timestamped_markers=self._timestamped_markers,
            )
            
        except Exception as e:
            self._logger.error(f"Error in BashAgent: {e}")
            if logging_dir:
                error_path = logging_dir / "error.txt"
                error_path.write_text(f"Error in BashAgent: {str(e)}")
            
            return AgentResult(
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
                timestamped_markers=self._timestamped_markers,
            )

    def _run_conversation_loop(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> None:
        """
        Run the main conversation loop with the Qwen model.
        
        The loop alternates between:
        1. Sending prompts/observations to the model
        2. Parsing tool calls from responses
        3. Executing one command in the terminal
        4. Collecting output for the next observation
        
        Loop terminates when:
        - Task is marked complete by the agent
        - No tool calls are proposed
        - Session ends
        - Max episodes reached
        """
        initial_prompt = instruction
        
        for episode in range(self._max_episodes):
            # Check if session is still alive before proceeding
            if not session.is_session_alive():
                self._logger.info("Session has ended, breaking out of conversation loop")
                break
            
            # Setup logging for this episode
            episode_logging_paths = self._setup_episode_logging(logging_dir, episode)
            
            # Get response from Qwen model
            if episode == 0:
                # First episode: send initial task instruction
                response = self._query_qwen_model(initial_prompt, logging_paths=episode_logging_paths, is_observation=False)
            else:
                # Subsequent episodes: send terminal output as observation
                # Limit output length to avoid token limits
                observation = self._limit_output_length(session.get_incremental_output())
                response = self._query_qwen_model(observation, logging_paths=episode_logging_paths, is_observation=True)
            
            # Add assistant response to conversation history
            self._conversation_history.append({"role": "assistant", "content": response})

            # Parse the response to extract tool calls
            parse_result = self._parse_qwen_response(response)
            
            # Record marker for asciinema debugging/visualization
            self._record_asciinema_marker(
                f"Episode {episode}: {len(parse_result.commands)} commands", session
            )
            
            # Handle parsing errors gracefully
            if parse_result.error:
                self._logger.warning(f"Parsing error in episode {episode}: {parse_result.error}")
                # Continue with empty commands to allow loop to terminate
                parse_result.commands = []
            
            # Check if agent proposed any tool calls
            # If no commands, agent may be done or stuck
            if not parse_result.commands:
                self.has_tool_call = False
                self._logger.warning(f"No tool calls proposed by agent (episode {episode})")
                break
            
            # Execute the parsed commands
            self._execute_commands(parse_result.commands, session)
            
            # Check for task completion
            if parse_result.is_task_complete:
                break

    def _setup_episode_logging(
        self, logging_dir: Path | None, episode: int
    ) -> Tuple[Path | None, Path | None, Path | None]:
        """
        Setup logging paths for an episode.
        
        Creates a subdirectory for the episode and returns paths for:
        - debug.json: Debug information (currently unused, reserved for future use)
        - prompt.txt: The prompt/observation sent to the model
        - response.txt: The raw response from the model
        
        Args:
            logging_dir: Base logging directory, or None to disable logging
            episode: Episode number for subdirectory naming
            
        Returns:
            Tuple of (debug_path, prompt_path, response_path), all None if logging disabled
        """
        if logging_dir is None:
            return None, None, None

        episode_logging_dir = logging_dir / f"episode-{episode}"
        episode_logging_dir.mkdir(parents=True, exist_ok=True)

        return (
            episode_logging_dir / "debug.json",
            episode_logging_dir / "prompt.txt",
            episode_logging_dir / "response.txt",
        )

    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """
        Limit output to specified byte length, keeping first and last portions.
        
        This prevents token limit issues when terminal output is very long.
        We preserve the beginning (most recent commands) and end (latest output)
        which are typically most relevant.
        
        Args:
            output: The terminal output string to limit
            max_bytes: Maximum byte length (default: 10000)
            
        Returns:
            Truncated output with middle portion replaced by indicator message
        """
        if len(output.encode("utf-8")) <= max_bytes:
            return output

        # Calculate portions (half each for first and last)
        # This ensures we keep both recent commands and latest output
        portion_size = max_bytes // 2
        output_bytes = output.encode("utf-8")

        # Get first and last portions
        # Use errors="ignore" to handle potential encoding edge cases
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        # Calculate omitted bytes for informative message
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )

        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    def _record_asciinema_marker(self, marker_text: str, session: TmuxSession) -> None:
        """
        Record a marker for asciinema debugging/visualization.
        
        Markers are timestamped events that can be used to annotate terminal
        recordings for easier debugging and analysis.
        
        Args:
            marker_text: Text description of the marker event
            session: Active tmux session to get timestamp from
        """
        current_timestamp = session.get_asciinema_timestamp()
        self._timestamped_markers.append((current_timestamp, marker_text))

    def _execute_commands(
        self,
        commands: List[QwenCommand],
        session: TmuxSession,
    ) -> None:
        """
        Execute a list of commands in the terminal session.
        
        Commands are sent sequentially with non-blocking execution.
        Each command uses its specified duration for timeout handling.
        
        Args:
            commands: List of QwenCommand objects to execute
            session: Active tmux session for command execution
        """
        for command in commands:
            try:
                # Send command with Enter key, non-blocking execution
                session.send_keys(
                    [command.command, "Enter"],
                    block=False,
                    min_timeout_sec=command.duration_sec,
                )
            except TimeoutError:
                self._logger.warning(f"Command timed out: {command.command}")

    def _query_qwen_model(
        self,
        prompt: str,
        logging_paths: Tuple[Path | None, Path | None, Path | None],
        is_observation: bool = False,
    ) -> str:
        """
        Query the Qwen model with the given prompt and handle logging.
        
        Wraps the actual API call with logging functionality to save
        prompts and responses for debugging/analysis.
        
        Args:
            prompt: The prompt or observation to send to the model
            logging_paths: Tuple of (debug_path, prompt_path, response_path)
            is_observation: If True, treat as tool output; if False, treat as user message
            
        Returns:
            The model's response content string
            
        Raises:
            Exception: If API call fails (re-raised from _call_qwen_model)
        """
        logging_path, prompt_path, response_path = logging_paths

        # Save prompt to file if logging enabled
        if prompt_path is not None:
            prompt_path.write_text(prompt)

        try:
            response = self._call_qwen_model(prompt, is_observation)
            
            # Save response to file if logging enabled
            if response_path is not None:
                response_path.write_text(response)
                
            return response
            
        except Exception as e:
            self._logger.error(f"Error querying Qwen model: {e}")
            raise

    def _call_qwen_model(self, prompt: str, is_observation: bool) -> str:
        """
        Call the Qwen model using OpenAI-compatible client.
        
        Args:
            prompt: The prompt or observation to send to the model
            is_observation: If True, treat as tool output; if False, treat as user message
            
        Returns:
            The model's response content string
            
        Raises:
            Exception: If API call fails or response content is None
        """
        # Prepare messages with conversation history
        # Note: This creates a reference, not a copy, so appending modifies the original
        # This is intentional to maintain conversation state
        messages = self._conversation_history
        
        self._logger.debug(f"Messages: {messages}")
        
        # Add the current prompt with appropriate role
        # Observations (terminal output) are sent as "tool" role
        # Initial instructions are sent as "user" role
        if is_observation:
            messages.append({"role": "tool", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # Update token usage statistics
            if hasattr(chat_response, 'usage') and chat_response.usage:
                self._total_input_tokens += chat_response.usage.prompt_tokens
                self._total_output_tokens += chat_response.usage.completion_tokens
            
            self._logger.debug(f"Response: {chat_response}")
                
            # Extract response content
            # vLLM and some Qwen deployments put content in reasoning_content instead of content
            message = chat_response.choices[0].message
            response_content = message.content
            
            # Fallback to reasoning_content if content is None (vLLM format)
            if response_content is None and hasattr(message, 'reasoning_content'):
                response_content = message.reasoning_content
                self._logger.debug(f"Using reasoning_content: {response_content}")
            else:
                self._logger.debug(f"Using content: {response_content}")
            
            if response_content is None:
                raise Exception("Model returned None response content in both content and reasoning_content")
            
            return response_content
            
        except Exception as e:
            raise Exception(f"Failed to call Qwen model: {e}")


    def _parse_qwen_response(self, response: str) -> QwenParseResult:
        """
        Parse the Qwen model response and extract commands and completion status.
        
        The response may contain tool calls in XML-like tags:
        <tool_call>{"name": "bash", "arguments": {"command": "..."}}</tool_call>
        
        Args:
            response: The raw response string from the model
            
        Returns:
            QwenParseResult with extracted commands and completion status
        """
        try:
            # Extract tool calls from the response content
            # Uses robust parsing to handle various formatting issues
            tool_calls = self._extract_tool_calls_from_reasoning(response)

            self._logger.debug(f"Tool calls: {tool_calls}")
            
            # Convert tool calls to QwenCommand objects
            # Only process "bash" tool calls, ignore others
            commands = []
            for tool_call in tool_calls:
                if tool_call.get("name") == "bash":
                    command = tool_call.get("arguments", {}).get("command", "")
                    if command:
                        commands.append(QwenCommand(command=command, duration_sec=self.command_duration_sec))
            
            # Check if agent indicated task completion
            is_task_complete = self._check_task_completion(response)
            self._logger.debug(f"Is task complete: {is_task_complete}")
            
            # If no commands extracted, assume task is complete
            # (agent may have finished or decided no more actions needed)
            if len(commands) == 0:
                is_task_complete = True
            
            return QwenParseResult(
                commands=commands,
                is_task_complete=is_task_complete,
                error="",
                warning=""
            )
                
        except Exception as e:
            # Return error result instead of raising to allow graceful handling
            return QwenParseResult(
                commands=[],
                is_task_complete=False,
                error=f"Unexpected error parsing Qwen response: {e}",
                warning=""
            )

    def _extract_tool_calls_from_reasoning(self, reasoning_content: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the reasoning content with robust parsing.
        
        Supports multiple formats and handles malformed JSON gracefully.
        The expected format is:
        <tool_call>
        {"name": "bash", "arguments": {"command": "pip install torch"}}
        </tool_call>
        
        Args:
            reasoning_content: The model's response text containing tool calls
            
        Returns:
            List of parsed tool call dictionaries with "name" and "arguments" keys
        """
        tool_calls = []
        
        if not reasoning_content or not isinstance(reasoning_content, str):
            self._logger.warning("Empty or invalid reasoning content")
            return tool_calls
        
        # Try multiple regex patterns in order of specificity
        # Start with strictest pattern, fall back to more flexible ones
        patterns = [
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',  # Standard format with complete JSON
            r'<tool_call>\s*(\{.*?)\s*</tool_call>',    # Missing closing brace in JSON
            r'<tool_call>\s*(.*?)\s*</tool_call>',      # Very flexible - any content
        ]
        
        matches = []
        for pattern in patterns:
            matches = re.findall(pattern, reasoning_content, re.DOTALL)
            if matches:
                self._logger.debug(f"Found {len(matches)} tool_call matches using pattern: {pattern}")
                break
        
        if not matches:
            self._logger.debug("No tool_call blocks found")
            return tool_calls
        
        # Parse each matched tool call
        for i, match in enumerate(matches):
            self._logger.debug(f"Tool call {i+1} raw match: {repr(match)}")
            
            # Clean up whitespace
            match = match.strip()
            if not match:
                self._logger.warning(f"Empty tool call {i+1}")
                continue
            
            # Try multiple parsing strategies (JSON, fixed JSON, regex, manual)
            tool_call = self._parse_single_tool_call(match, i+1)
            if tool_call:
                tool_calls.append(tool_call)
        
        self._logger.debug(f"Final tool calls: {tool_calls}")
        return tool_calls
    
    def _parse_single_tool_call(self, match: str, call_index: int) -> Optional[Dict[str, Any]]:
        """
        Parse a single tool call with multiple fallback strategies.
        
        Uses a cascading approach: tries strict parsing first, then progressively
        more lenient methods. This handles various formatting issues that can occur
        in model outputs.
        
        Args:
            match: The raw string match from regex extraction
            call_index: Index of the tool call (for logging)
            
        Returns:
            Parsed tool call dict with "name" and "arguments", or None if all strategies fail
        """
        # Strategy 1: Direct JSON parsing (most common case)
        try:
            tool_call = json.loads(match)
            if self._validate_tool_call(tool_call):
                self._logger.debug(f"Successfully parsed tool call {call_index} using JSON: {tool_call}")
                return tool_call
        except json.JSONDecodeError as e:
            self._logger.debug(f"JSON parse failed for tool call {call_index}: {e}")
        
        # Strategy 2: Fix common JSON issues and retry
        # Handles trailing commas, unescaped quotes, etc.
        try:
            fixed_match = self._fix_json_issues(match)
            if fixed_match != match:
                tool_call = json.loads(fixed_match)
                if self._validate_tool_call(tool_call):
                    self._logger.debug(f"Successfully parsed tool call {call_index} using fixed JSON: {tool_call}")
                    return tool_call
        except (json.JSONDecodeError, Exception) as e:
            self._logger.debug(f"Fixed JSON parse failed for tool call {call_index}: {e}")
        
        # Strategy 3: Extract command using regex (when JSON structure is broken)
        # Assumes "bash" tool and extracts command directly
        try:
            command = self._extract_command_with_regex(match)
            if command:
                tool_call = {
                    "name": "bash",
                    "arguments": {
                        "command": command
                    }
                }
                self._logger.debug(f"Successfully extracted tool call {call_index} using regex: {tool_call}")
                return tool_call
        except Exception as e:
            self._logger.debug(f"Regex extraction failed for tool call {call_index}: {e}")
        
        # Strategy 4: Manual parsing for severely malformed JSON
        # Uses regex to find name and command fields independently
        try:
            tool_call = self._manual_parse_tool_call(match)
            if tool_call:
                self._logger.debug(f"Successfully parsed tool call {call_index} using manual parsing: {tool_call}")
                return tool_call
        except Exception as e:
            self._logger.debug(f"Manual parsing failed for tool call {call_index}: {e}")
        
        # All strategies failed
        self._logger.error(f"All parsing strategies failed for tool call {call_index}")
        return None
    
    def _validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validate that a tool call has the required structure.
        
        Checks that the tool call is a dict with:
        - "name" field (string)
        - "arguments" field (dict)
        - "arguments.command" field (string)
        
        Args:
            tool_call: The parsed tool call dictionary to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        if not isinstance(tool_call, dict):
            return False
        
        if "name" not in tool_call or "arguments" not in tool_call:
            return False
        
        if not isinstance(tool_call["arguments"], dict):
            return False
        
        if "command" not in tool_call["arguments"]:
            return False
        
        if not isinstance(tool_call["arguments"]["command"], str):
            return False
        
        return True
    
    def _fix_json_issues(self, match: str) -> str:
        """
        Fix common JSON formatting issues that can occur in model outputs.
        
        Handles:
        - Trailing commas before closing braces/brackets
        - Unescaped quotes in string values
        
        Args:
            match: The JSON string to fix
            
        Returns:
            Fixed JSON string (may be unchanged if no issues found)
        """
        # Remove trailing commas before closing braces/brackets
        # JSON doesn't allow trailing commas, but models sometimes generate them
        match = re.sub(r',\s*}', '}', match)
        match = re.sub(r',\s*]', ']', match)
        
        # Fix unescaped quotes in command values
        # Commands may contain quotes that need escaping
        def escape_command_quotes(m):
            command = m.group(1)
            # Only escape quotes that aren't already escaped
            escaped_command = re.sub(r'(?<!\\)"', r'\\"', command)
            return f'"command": "{escaped_command}"'
        
        # Apply the fix to command field values
        fixed = re.sub(r'"command":\s*"([^"]*(?:\\.[^"]*)*)"', escape_command_quotes, match)
        
        return fixed
    
    def _extract_command_with_regex(self, match: str) -> Optional[str]:
        """
        Extract command string using regex when JSON parsing fails.
        
        This is a fallback method that directly extracts the command value
        from malformed JSON by finding the "command" field and its value.
        
        Args:
            match: The malformed JSON string
            
        Returns:
            Extracted command string, or None if not found
        """
        # Find the "command" field and extract its value
        # Pattern: "command": "..."
        start_pattern = r'"command":\s*"'
        start_match = re.search(start_pattern, match)
        
        if start_match:
            start_pos = start_match.end()
            # Find the closing quote
            # Note: This is a simplified approach - finds the last quote before closing brace
            # May fail if command contains unmatched quotes, but handles most cases
            end_pos = match.rfind('"', start_pos)
            if end_pos > start_pos:
                command = match[start_pos:end_pos]
                # Unescape any escaped quotes in the command
                command = command.replace('\\"', '"')
                if command.strip():
                    return command.strip()
        
        return None
    
    def _manual_parse_tool_call(self, match: str) -> Optional[Dict[str, Any]]:
        """
        Manually parse tool call for severely malformed JSON.
        
        This is the last-resort parsing strategy. It uses regex to find
        the "name" and "command" fields independently, then reconstructs
        the tool call structure.
        
        Args:
            match: The malformed JSON string to parse
            
        Returns:
            Reconstructed tool call dict, or None if parsing fails
        """
        # Look for key patterns using regex
        # Extract name and command fields independently
        name_match = re.search(r'"name":\s*"([^"]*)"', match)
        command_match = re.search(r'"command":\s*"([^"]*(?:\\.[^"]*)*)"', match)
        
        if name_match and command_match:
            name = name_match.group(1)
            # Unescape quotes in command
            command = command_match.group(1).replace('\\"', '"')
            
            return {
                "name": name,
                "arguments": {
                    "command": command
                }
            }
        
        return None

    def _check_task_completion(self, reasoning_content: str) -> bool:
        """
        Check if the task is marked as complete in the reasoning content.
        
        Looks for various completion indicators that the model might use
        to signal that the task is finished.
        
        Args:
            reasoning_content: The model's response text to check
            
        Returns:
            True if any completion indicator is found, False otherwise
        """
        # List of phrases that indicate task completion
        # Covers various formats the model might use
        completion_indicators = [
            "task_complete: true",
            "task_complete:true", 
            "task complete: true",
            "task complete:true",
            "task is complete",
            "task completed",
            "finished the task",
            "task finished"
        ]
        
        reasoning_lower = reasoning_content.lower()
        return any(indicator in reasoning_lower for indicator in completion_indicators)
