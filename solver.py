import numpy as np
from numpy import ndarray

class solverSudoku():
    def __init__(self, board: list):
        # Initialize the Sudoku solver with a 2D array (9x9) representing the board
        self.board = [row[:] for row in board]  # Deep copy
        # self.original_board = [row[:] for row in grid]
        self.steps = 0
        self.backtracks = 0

    def is_valid(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        for j in range(9):
            if self.board[row][j] == num:
                return False
                
        # Check column
        for i in range(9):
            if self.board[i][col] == num:
                return False
                
        # Check 3x3 box
        box_row = 3 * (row // 3)
        box_col = 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if self.board[i][j] == num:
                    return False
                    
        return True
    
    def find_empty(self) -> tuple[int, int] | None:
        """Find empty cell, returns None if no empty cell found"""
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    
    def initialize_possibilities(self) -> list[list[set[int]]]:
        """Initialize possibility sets for each empty cell"""
        possibilities = [[set() for _ in range(9)] for _ in range(9)]
        
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    for num in range(1, 10):
                        if self.is_valid(i, j, num):
                            possibilities[i][j].add(num)
                            
        return possibilities
    
    def update_possibilities(self, possibilities: list[list[set[int]]], 
                           row: int, col: int, num: int) -> None:
        """Update possibilities after placing a number"""
        # Remove from row
        for j in range(9):
            possibilities[row][j].discard(num)
            
        # Remove from column
        for i in range(9):
            possibilities[i][col].discard(num)
            
        # Remove from 3x3 box
        box_row = 3 * (row // 3)
        box_col = 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                possibilities[i][j].discard(num)
                
        # Clear the filled cell
        possibilities[row][col].clear()
    
    def solve_greedy(self) -> bool:
        """Solve using greedy constraint propagation"""
        possibilities = self.initialize_possibilities()
        progress = True
        
        while progress:
            progress = False
            
            # Naked Singles: cells with only one possibility
            for i in range(9):
                for j in range(9):
                    if self.board[i][j] == 0 and len(possibilities[i][j]) == 1:
                        num = next(iter(possibilities[i][j]))
                        self.board[i][j] = num
                        self.update_possibilities(possibilities, i, j, num)
                        progress = True
                        self.steps += 1
                        
            # Hidden Singles: numbers that can only go in one place
            # Check rows
            for i in range(9):
                for num in range(1, 10):
                    candidates = []
                    for j in range(9):
                        if self.board[i][j] == 0 and num in possibilities[i][j]:
                            candidates.append(j)
                    if len(candidates) == 1:
                        j = candidates[0]
                        self.board[i][j] = num
                        self.update_possibilities(possibilities, i, j, num)
                        progress = True
                        self.steps += 1
                        
            # Check columns
            for j in range(9):
                for num in range(1, 10):
                    candidates = []
                    for i in range(9):
                        if self.board[i][j] == 0 and num in possibilities[i][j]:
                            candidates.append(i)
                    if len(candidates) == 1:
                        i = candidates[0]
                        self.board[i][j] = num
                        self.update_possibilities(possibilities, i, j, num)
                        progress = True
                        self.steps += 1
                        
            # Check 3x3 boxes
            for box in range(9):
                start_row = 3 * (box // 3)
                start_col = 3 * (box % 3)
                
                for num in range(1, 10):
                    candidates = []
                    for i in range(start_row, start_row + 3):
                        for j in range(start_col, start_col + 3):
                            if self.board[i][j] == 0 and num in possibilities[i][j]:
                                candidates.append((i, j))
                    if len(candidates) == 1:
                        i, j = candidates[0]
                        self.board[i][j] = num
                        self.update_possibilities(possibilities, i, j, num)
                        progress = True
                        self.steps += 1
                        
        # Check if solved
        return all(self.board[i][j] != 0 for i in range(9) for j in range(9))
    
    def find_most_constrained_cell(self) -> tuple[int, int] | None:
        """Find cell with minimum possibilities (MCV heuristic)"""
        min_possibilities = 10
        best_cell = None
        
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    possibilities = sum(1 for num in range(1, 10) if self.is_valid(i, j, num))
                    if possibilities < min_possibilities:
                        min_possibilities = possibilities
                        best_cell = (i, j)
                        
        return best_cell
    
    def solve_smart_backtracking(self) -> bool:
        """Enhanced backtracking with MCV heuristic"""
        self.steps += 1
        cell = self.find_most_constrained_cell()
        
        if cell is None:
            return True  # Solved
            
        row, col = cell
        
        # Try numbers 1-9
        for num in range(1, 10):
            if self.is_valid(row, col, num):
                self.board[row][col] = num
                
                if self.solve_smart_backtracking():
                    return True
                    
                self.board[row][col] = 0
                self.backtracks += 1
                
        return False
    
    def solve_combined(self) -> bool:
        """Combined greedy + backtracking approach"""
        # First try greedy approach
        if self.solve_greedy():
            return True
            
        # If greedy couldn't solve completely, use backtracking
        return self.solve_smart_backtracking()
    
    def getFinishedBoard(self) -> ndarray:
        """
        Returns the finished Sudoku board as a 9x9 numpy array.
        """
        
        board = self.board
        newBoard = []
        
        # Flatten the board into a 1D list for easy manipulation
        for i in range(len(board)):
            for j in range(len(board)):
                newBoard.append(board[i][j])

        # Reshape the list back into a 9x9 array
        newBoard = np.array(newBoard).reshape(9,9)

        return newBoard