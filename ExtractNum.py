from itertools import product
from solver import solverSudoku
import numpy as np
from numpy import ndarray


def allTable(allGrid: list) -> list:
    numbers = []                # List to store the extracted numbers
    rangeAll = range(0, 9)     # Range for rows and columns (0 to 8)

    # Iterate over all combinations of index, row, and column using Cartesian product
    for index, row, column in product(rangeAll, rangeAll, rangeAll):
        # Define attributes to locate the specific cell in the HTML structure
        attr = {
            'class': 'game-grid__cell', 
            'data-row': f'{row}',
            'data-column': f'{column}'        
        }

        try:
            cell = allGrid[index].findChild('div', attrs=attr)
            # Find the SVG element representing the number within the cell
            number = cell.findChild('svg', attrs={'class', 'default'})

            if number != None:
                # Extract text from the SVG, remove newline characters
                number = number.get_text().replace('\n', "")
            else:
                # If no number is present, denote the cell as empty (0)
                number = '0'
            
            numbers.append(int(number))
        except AttributeError:
            # Handle cases where the cell or number is not found
            if (index + 1) % 3 == 0:
                column = 0
            column += 1
    
    return numbers



def __organizedTable(allNumbers: list) -> ndarray:
    # allNumbers (list): A flat list of 81 integers representing the Sudoku board.
    
    quadrant = [allNumbers[0:9]]  # Initialize with the first row
    qdLine, newqdLine = [], []   # Lists to store quadrants and the final organized lines

    # Split the flat list into rows and append to the quadrant list
    for i in range(9, 81, 9):
        line = [allNumbers[i:i+9]]
        quadrant = np.append(quadrant, line, axis=0)
        
    # Extract 3x3 boxes (quadrants) from the rows
    for row in range(0, 9, 3):
        for column in range(0, 9, 3):
            # Extract a 3x3 block and append to qdLine
            qdLine.append(quadrant[row:row+3, column:column+3])
    
    # Organize the numbers by traversing each box row-wise
    for box in range(0, 9):
        for row in range(0, 3):
            for column in range(0, 3):
                newqdLine.append(qdLine[box][row][column])
    
    # Convert the organized list into a 9x9 NumPy array
    newqdLineArray = np.array(newqdLine).reshape(9, 9)

    return newqdLineArray



def finishedTable(numbers: list) -> ndarray:
    newBoard = []  # List to store the final solved board

    # Initialize the Sudoku solver with the organized board
    solver = solverSudoku(__organizedTable(numbers))
    # Attempt to solve the Sudoku puzzle
    solver.solve_combined()
    
    # Retrieve the solved board as a NumPy array
    finishedBoard = solver.getFinishedBoard()

    # Split the solved board into 3x3 boxes for further processing if needed
    for row in range(0, 9, 3):
        for column in range(0, 9, 3):
            # Append each 3x3 block to newBoard
            newBoard.append(finishedBoard[row:row+3, column:column+3])
    
    # Reshape the list of 3x3 blocks back into a 9x9 grid and convert to a Python list
    newBoard = np.array(newBoard).reshape(9, 9).tolist()

    return newBoard
