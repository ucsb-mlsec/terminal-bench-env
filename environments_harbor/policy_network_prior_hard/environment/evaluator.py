#!/usr/bin/env python3

class PositionEvaluator:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.win_length = 4
    
    def evaluate(self, board, player):
        """
        Evaluate all possible moves for a given board position.
        
        Args:
            board: 6x7 2D list/array with values 0 (empty), 1 (player 1), 2 (player 2)
            player: Current player number (1 or 2)
        
        Returns:
            Dictionary mapping column indices to float scores
        """
        scores = {}
        opponent = 3 - player  # If player is 1, opponent is 2; if player is 2, opponent is 1
        
        for col in range(self.cols):
            if not self.is_valid_move(board, col):
                continue
            
            row = self.get_drop_row(board, col)
            score = 0.0
            
            # Check if this move wins immediately
            if self.check_win_at_position(board, row, col, player):
                score = 1000.0
                scores[col] = score
                continue
            
            # Check if this move blocks an opponent win
            # Simulate opponent move
            if self.check_win_at_position(board, row, col, opponent):
                score = 500.0
                scores[col] = score
                continue
            
            # Check if this move would give opponent a win on next turn
            # (playing here would allow opponent to win directly above)
            if row > 0:  # There's a row above
                if self.check_win_at_position(board, row - 1, col, opponent):
                    score = -100.0
                    scores[col] = score
                    continue
            
            # Center column preference
            center_distance = abs(col - 3)
            score += (7 - center_distance) * 2
            
            # Evaluate threats created by this move
            score += self.count_threats(board, row, col, player) * 10
            
            # Penalize moves that create threats for opponent
            score -= self.count_threats(board, row, col, opponent) * 5
            
            # Bonus for moves that create multiple open-ended opportunities
            score += self.count_open_sequences(board, row, col, player) * 5
            
            scores[col] = score
        
        return scores
    
    def is_valid_move(self, board, column):
        """Check if a column isn't full."""
        if column < 0 or column >= self.cols:
            return False
        return board[0][column] == 0
    
    def get_drop_row(self, board, column):
        """Return the row where a piece would land in the given column."""
        for row in range(self.rows - 1, -1, -1):
            if board[row][column] == 0:
                return row
        return -1
    
    def check_win_at_position(self, board, row, col, player):
        """Check if placing a piece at (row, col) would create 4-in-a-row for player."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        
        # Temporarily place the piece
        original = board[row][col]
        board[row][col] = player
        
        # Check all four directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal-right, diagonal-left
        
        for dr, dc in directions:
            count = 1  # Count the piece we just placed
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= self.win_length:
                board[row][col] = original
                return True
        
        board[row][col] = original
        return False
    
    def count_threats(self, board, row, col, player):
        """Count the number of 3-in-a-row threats created by placing at (row, col)."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return 0
        
        original = board[row][col]
        board[row][col] = player
        
        threats = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            empty_ends = 0
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if board[r][c] == player:
                    count += 1
                    r += dr
                    c += dc
                elif board[r][c] == 0:
                    empty_ends += 1
                    break
                else:
                    break
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if board[r][c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                elif board[r][c] == 0:
                    empty_ends += 1
                    break
                else:
                    break
            
            # A threat is 3 pieces with at least one open end
            if count == 3 and empty_ends > 0:
                threats += 1
        
        board[row][col] = original
        return threats
    
    def count_open_sequences(self, board, row, col, player):
        """Count sequences of 2+ pieces with open ends."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return 0
        
        original = board[row][col]
        board[row][col] = player
        
        sequences = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            open_ends = 0
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if board[r][c] == player:
                    count += 1
                    r += dr
                    c += dc
                elif board[r][c] == 0:
                    open_ends += 1
                    break
                else:
                    break
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if board[r][c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                elif board[r][c] == 0:
                    open_ends += 1
                    break
                else:
                    break
            
            # Reward sequences with both ends open
            if count >= 2 and open_ends == 2:
                sequences += count
        
        board[row][col] = original
        return sequences