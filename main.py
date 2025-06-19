from wSelenium import Navigator
from ExtractNum import allTable, finishedTable

if __name__ == "__main__":
    navigator = Navigator()
    
    bs = navigator.BeautifulSoup()
    driver = navigator.GetDriver()
    
    board = bs.findAll('div', attrs={'class':'game-grid__group'})
    allNumbers = allTable(board)
    finishedBoard = finishedTable(allNumbers)

    navigator.FillCells(finishedBoard)