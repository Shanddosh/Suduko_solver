from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium import webdriver
from bs4 import BeautifulSoup

import sys
import os 


LINK = "https://sudokutable.com" 

HEADER = {
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 OPR/80.0.4170.72'
}

# Configure Chrome WebDriver options
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Suppress logging
options.add_experimental_option("detach", True)  # Keep the browser open after script finishes


def resource_path(path: str) -> str:
    """
    Returns the absolute path to the resource, handling PyInstaller's temporary folder.

    Parameters:
        path (str): Relative path to the resource.

    Returns:
        str: Absolute path to the resource.
    """
    try:
        # If the script is bundled by PyInstaller, use the temporary folder
        base_path = sys._MEIPASS
    except Exception:
        # Otherwise, use the directory of the current script
        base_path = os.path.dirname(__file__)
    
    return os.path.join(base_path, path)


class Navigator:
    def __init__(self):
        """
        Initializes the Navigator by setting up the WebDriver, navigating to the Sudoku website,
        and setting the difficulty level.
        """
        
        
        self.driver = webdriver.Chrome(options=options)
        self.actions = ActionChains(self.driver) # Initialize ActionChains for advanced user interactions
        self.wait = WebDriverWait(self.driver, 30) 

        # Navigate to the Sudoku website
        self.driver.get(LINK)
        self.driver.set_window_size(1100, 1000)

        # Wait until the "Agree Cookies" button is clickable, then click it to accept cookies
        self.wait.until(EC.element_to_be_clickable((By.ID, "agree_cookies")))
        self.driver.find_element(by=By.ID, value='agree_cookies').click()

        # Click on the difficulty selection dropdown
        self.driver.find_element(by=By.ID, value='difficulty').click()

        # Select the desired difficulty level (e.g., 'Hard') from the dropdown
        # XPath indexing starts at 1, so li[4] selects the fourth option
        self.driver.find_element(by=By.XPATH, value='//*[@id="difficulty"]/ul/li[4]').click()

        # Store the current page source for parsing
        self.url = self.driver.page_source

    def BeautifulSoup(self):
        """
        Parses the current page source using BeautifulSoup.

        Returns:
            BeautifulSoup: Parsed HTML of the current page.
        """
        
        bs = BeautifulSoup(self.url, 'html.parser')
        return bs
    
    def GetDriver(self):
        """
        Provides access to the WebDriver instance.

        Returns:
            WebDriver: The current Selenium WebDriver instance.
        """
        
        return self.driver

    def FillCells(self, ListNum: list):
        """
        Fills the Sudoku grid on the website with the provided numbers.

        Parameters:
            ListNum (list): A list of lists containing numbers to fill in the Sudoku grid.
                            Each sublist represents a 3x3 box in the grid.
        """
        
        
        bs = self.BeautifulSoup()  # Parse the current page source
        driver = self.GetDriver()   # Get the WebDriver instance

        print("Solving...")
        
        # Iterate through each of the 9 3x3 boxes in the Sudoku grid
        for box in range(0, 9): 
            alreadyHere = []           # List to keep track of numbers already present in the box
            resultList = ListNum[box]  # Numbers to be filled in the current box
            
            print(f"{box + 1} box - {resultList}")
            # Iterate through each of the 9 cells within the current box
            for cell in range(0, 9):
                # Locate the current box in the HTML structure
                group = bs.findAll('div', attrs={'class': 'game-grid__group'})[box]
                
                # Locate the specific cell within the current box
                ggCell = group.findAll('div', attrs={'class': 'game-grid__cell'})[cell]
                
                # Find all SVG elements within the current cell that represent empty cells
                allCell = driver.find_elements(
                    by=By.XPATH, 
                    value=f"//div[@class='game-grid__group'][{box+1}]/div/*[name()='svg' and @class='']"
                )

                # Check if the current cell is empty by inspecting the 'class' attribute of the SVG
                cellEmpty = ggCell.find('svg')['class']

                if len(cellEmpty) > 0:
                    # If the cell is not empty, extract the number present
                    numToAppend = ggCell.find('svg', attrs={'class', 'default'}).get_text().replace('\n', '')
                    alreadyHere.append(int(numToAppend))
                else:
                    # If the cell is empty, continue to the next cell
                    continue
            
            # Remove numbers that are already present in the box from the list of numbers to fill
            for i in alreadyHere:
                if i in resultList:
                    resultList.remove(i)

            # Iterate through each empty cell in the current box to fill it with the appropriate number
            for box_index in range(len(allCell)):
                for num in resultList:
                    try:
                        # Move to the cell, click it, and send the number key
                        self.actions.move_to_element(allCell[box_index]).click().send_keys(num).perform()
                        box_index += 1  # Move to the next cell
                    except IndexError:
                        # If there are no more cells to fill, exit the loop
                        break
                break  # Exit after filling the first box (adjust logic if needed)

