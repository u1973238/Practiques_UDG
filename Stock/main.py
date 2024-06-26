
#import requests
#from bs4 import BeautifulSoup
#response = requests.get("https://en.wikipedia.org/wiki/Web_scraping")
#bs = BeautifulSoup(response.text,"lxml")
#print(bs.find("p").text)

import requests
from bs4 import BeautifulSoup

# Obtenir la pàgina web
response = requests.get("https://www.nytimes.com/international/section/business/economy")

# Utilitzar BeautifulSoup amb el parser lxml
bs = BeautifulSoup(response.text, "lxml")

# Trobar tots els paràgrafs
paragraphs = bs.find_all('article')

# Iterar sobre cada paràgraf i imprimir el seu text
for paragraph in paragraphs:
    print(paragraph.text)
