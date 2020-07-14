
from bs4 import BeautifulSoup
import urllib.request

def searchSpider():
    #opening URL
    url = "https://en.wikipedia.org/wiki/Google"
    sourceCode = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_code, "html.parser")

    body = soup.find('div', {'class': 'mw-parser-output'})
    file.write(str(body.text))
    search = input('type "q" to exit')
    if search == 'q' or search == 'Q':
        print("Quit")
        exit()
    else:
        print("Create text file")
        file = open('input.txt')
        searchSpider()



