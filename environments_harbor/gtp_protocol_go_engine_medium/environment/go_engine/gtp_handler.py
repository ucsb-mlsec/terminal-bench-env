#!/usr/bin/env python3

class GTPHandler:
    """
    GTP (Go Text Protocol) command handler.

    This module should implement the GTP protocol for communicating with a Go engine.
    The GTP protocol is a text-based protocol where:
    - Commands are sent as text
    - Success responses start with '=' followed by optional data, ending with two newlines
    - Error responses start with '?' followed by an error message, ending with two newlines
    - Commands may optionally include a numeric ID prefix

    Required commands to implement:
    - protocol_version: Return the GTP protocol version (2)
    - name: Return the engine name
    - version: Return the engine version
    - known_command <cmd>: Return 'true' if command is supported, 'false' otherwise
    - list_commands: List all supported commands
    - boardsize <size>: Set board size (only 9, 13, or 19 are valid)
    - clear_board: Reset the board
    - play <color> <vertex>: Place a stone (validate color and coordinates)
    - genmove <color>: Generate a move
    - showboard: Display the current board state
    - quit: Exit gracefully
    """

    def __init__(self):
        # TODO: Initialize the GTP handler
        # You may want to create a GoEngine instance here
        pass

    def handle_command(self, command_string):
        """
        Parse and execute a GTP command, returning a properly formatted response.

        Args:
            command_string: A GTP command string, possibly with a numeric ID prefix

        Returns:
            A formatted GTP response string
        """
        # TODO: Implement command parsing and dispatching
        pass
