#!/usr/bin/env python3

from go_engine.engine import GoEngine

class GTPHandler:
    """GTP (Go Text Protocol) command handler with intentional bugs"""
    
    def __init__(self, engine):
        self.engine = engine
        self.commands = {
            'protocol_version': self.protocol_version,
            'name': self.name,
            'version': self.version,
            'known_command': self.known_command,
            'list_commands': self.list_commands,
            'boardsize': self.boardsize,
            'clear_board': self.clear_board,
            'play': self.play,
            'genmove': self.genmove,
            'showboard': self.showboard,
            'quit': self.quit_command,
        }
    
    def parse_command(self, command_string):
        """Parse GTP command string into components"""
        parts = command_string.strip().split()
        if not parts:
            return None, None, []
        
        # Check if first part is a command ID (numeric)
        command_id = None
        if parts[0].isdigit():
            command_id = parts[0]
            parts = parts[1:]
        
        if not parts:
            return command_id, None, []
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        return command_id, command, args
    
    def format_response(self, success, message='', command_id=None):
        """Format GTP response - BUGGY: missing proper formatting"""
        # BUG: Not including command_id properly
        if success:
            response = '=' + message
        else:
            response = '?' + message
        # BUG: Missing double newline terminator
        return response + '\n'
    
    def process_command(self, command_string):
        """Process a GTP command and return response"""
        command_id, command, args = self.parse_command(command_string)
        
        if command is None:
            return self.format_response(False, 'empty command')
        
        if command not in self.commands:
            return self.format_response(False, f'unknown command: {command}', command_id)
        
        try:
            handler = self.commands[command]
            result = handler(*args)
            return result
        except Exception as e:
            return self.format_response(False, str(e), command_id)
    
    def protocol_version(self):
        """Return GTP protocol version - BUGGY"""
        # BUG: Wrong format, missing proper response structure
        return '2\n'
    
    def name(self):
        """Return engine name - BUGGY"""
        # BUG: Missing '=' prefix
        return 'SimpleGoEngine\n'
    
    def version(self):
        """Return engine version - BUGGY"""
        # BUG: Incorrect response format
        return '= 1.0'
    
    def known_command(self, cmd):
        """Check if command is known - BUGGY"""
        # BUG: Logic is inverted
        if cmd in self.commands:
            return self.format_response(True, 'false')
        else:
            return self.format_response(True, 'true')
    
    def list_commands(self):
        """List all known commands - BUGGY"""
        # BUG: Missing some commands and wrong format
        cmd_list = ['protocol_version', 'name', 'version', 'boardsize', 'quit']
        return '= ' + '\n'.join(cmd_list)
    
    def boardsize(self, size):
        """Set board size - BUGGY: No validation"""
        # BUG: Missing validation for 9, 13, 19 only
        try:
            size = int(size)
            self.engine.set_board_size(size)
            return self.format_response(True)
        except:
            return self.format_response(False, 'invalid size')
    
    def clear_board(self):
        """Clear the board - BUGGY format"""
        self.engine.clear_board()
        # BUG: Wrong response format
        return '=\n'
    
    def play(self, color, vertex):
        """Play a move - BUGGY: Missing validation and capture detection"""
        # BUG: No color validation
        # BUG: No coordinate validation
        # BUG: Not calling remove_captured_stones()
        try:
            col_letter = vertex[0].upper()
            row = int(vertex[1:])
            
            # Convert to 0-indexed
            col = ord(col_letter) - ord('A')
            if col >= 8:  # Skip 'I'
                col -= 1
            row = self.engine.board_size - row
            
            self.engine.place_stone(row, col, color.upper())
            return self.format_response(True)
        except:
            return self.format_response(False, 'invalid move')
    
    def genmove(self, color):
        """Generate a move - INCOMPLETE"""
        # TODO: FIXME - This is incomplete
        # BUG: Very basic implementation, doesn't validate color
        try:
            move = self.engine.generate_move(color.upper())
            if move is None:
                return self.format_response(True, 'pass')
            
            row, col = move
            # Convert to GTP coordinates
            if col >= 8:
                col_letter = chr(col + ord('A') + 1)
            else:
                col_letter = chr(col + ord('A'))
            row_num = self.engine.board_size - row
            
            # BUG: Wrong response format
            return '= ' + col_letter + str(row_num)
        except:
            return self.format_response(False, 'cannot generate move')
    
    def showboard(self):
        """Show current board - MISSING/INCOMPLETE"""
        # FIXME: Not implemented properly
        return self.format_response(True, 'showboard not fully implemented')
    
    def quit_command(self):
        """Quit the engine - BUGGY"""
        # BUG: Missing proper format
        return '=\n'