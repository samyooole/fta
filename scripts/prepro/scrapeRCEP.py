from bs4 import BeautifulSoup
import requests
import re
import pdfx
import fitz
import os


def linkClicker(link):
    page = requests.get(link)
    f = open("temp2.pdf", "wb")                                              # I don't quite like this move - could be more elegant
    f.write(page.content)                                        # We write it off as a pdf then read it again, because pypdf2 does not (?) have an direct parser. To find and improve.
    f.close()

    try:
        #pdf = textract.process('temp2.pdf', method = 'pdfminer', encoding = 'ascii')
        with fitz.open("temp2.pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        os.remove('temp2.pdf')
        return text
    except:
        print(link + " is probably a scanned PDF, no text returned")
        os.remove('temp2.pdf')
        return None

motherpdf = fitz.open('temp.pdf')

prefix = '/non-financial-assistance/for-singapore-companies/free-trade-agreements/ftas/singapore-ftas/'
fta_name = 'rcep'
os.chdir('text') # will probably throw an error on loop
try:
    os.makedirs('rcep')
except:
    print("dir alr made")
os.chdir('rcep')

for page in motherpdf.pages():
    linklist = page.get_links()
    for item in linklist:
        url = item['uri']
        
        if re.search('terms-of-use', url.lower()) is not None :
            continue # I don't want to go to the terms of use website

        text = linkClicker(url)

        sub_fta = url


        if re.search('terms-of-use', sub_fta.lower()) is not None :
            continue # I don't want to go to the terms of use website
        
        sub_fta_text = linkClicker(sub_fta)
        #simple cleaning: NOTE should try to be less brute force? very manual rn based on observations
        
        #try:
        #    sub_fta_text = str.replace(sub_fta_text, "\n", "")
        #except:
        #    continue
        #sub_fta_text = str.replace(sub_fta_text, "™", "'")
        #finding a name
        sub_fta_name = re.sub(pattern = 'rcep', repl = '', string = sub_fta, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "https", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "http", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "www.enterprisesg.gov.sg", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "media", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "esg", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "files", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "non-financial-assistance", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "for-companies", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = "free-trade-agreements", repl = '', string = sub_fta_name, flags = re.IGNORECASE)
        sub_fta_name = re.sub(pattern = 'legal-text', repl = '', string = sub_fta_name, flags=re.IGNORECASE)
        sub_fta_name = re.sub(pattern = '.pdf', repl = '', string = sub_fta_name)
        sub_fta_name = re.sub(pattern = '/', repl = '', string = sub_fta_name)
        sub_fta_name = re.sub(pattern = ':', repl = '', string = sub_fta_name)

        #dump as txt file
        f = open(sub_fta_name + ".txt", "w")
        f.write(sub_fta_text)
        f.close()
        