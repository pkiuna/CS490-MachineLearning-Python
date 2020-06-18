from bs4 import BeautifulSoup #libraries
import urllib.request
import os


def search_spider(sea, lim):
    url = "https://en.wikipedia.org/w/index.php?limit=" + lim + "&offset=0&search=" + sea #search key word
    source_code = urllib.request.urlopen(url) #open the url to source code
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser") #parse using BeautfulSoap for html
    # parse to variable
    result_list = soup.find_all('div', {'class': "mw-search-result-heading"}) #include div with this class in the result
    print(soup.title.string)
    print(result_list)

    for div in result_list:
        link = div.find('a')
        href = "https://en.wikipedia.org" + link.get('href')  #analyzing the first result
        if link.get('href').startswith("http"):
            href = link.get('href')
        get_data(href)
        get_data_links(href)

    # print out the title and links of the result page
    for div in result_list:
        title_all = div.find('a')
        title = title_all.get('title')
        links = "https://en.wikipedia.org" + title_all.get('href')
        print("titles of the search result page is " + title)
        print("links of the search result page is " + links)
        file_title.write(str(title))


# sample: parse text to file
def get_data(url):
    source_code = urllib.request.urlopen(url)
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser")
    body = soup.find('div', {'class': 'mw-parser-output'})
    file2.write(str(body.text))


#  get all links in the page
def get_data_links(url):
    source_code = urllib.request.urlopen(url)
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser")
    #links to body
    body = soup.find_all('a')
    for a in body:
        # return the link using 'href' using get
        links_all = a.get('href')
        title_page = a.get('title')
        # print(links_all)
        file_link.write(str(links_all))
        file_title.write(str(title_page))


# input deep_learning
search = input('type something you want to search in wiki: ')
limit = input('What is the number of results you want to get back?: ')

if not os.path.exists(search):
    print("Creating file " + search)
    file2 = open(search + '.txt', 'a+', encoding='utf-8')
    file_title = open(search + '_title.txt', 'a+', encoding='utf-8')
    file_link = open(search + '_link.txt', 'a+', encoding='utf-8')

search_spider(search, limit)