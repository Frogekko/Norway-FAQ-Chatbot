# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:35:05 2025

@author: Fredrik

Used this webpage for help: https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://www.reddit.com/r/norge/"
https = urlopen(url).read()
soup = BeautifulSoup(https, features="html.parser")
    
text = soup.body.get_text(separator="\n", strip=True)

with open("soup_of_words.txt", "w") as file:
    file.write(text)
    print("Text file written")