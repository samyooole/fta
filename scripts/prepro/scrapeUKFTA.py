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

def bruteCleaner(text, name):
    #try:
    #    text = str.replace(text, "\n", "")
    #except:
    #    return "I don't know how to clean this"
    #text = str.replace(text, "™", "'")

    #dump as txt file
    f = open(name + ".txt", "w")
    f.write(text)
    f.close()

link_1 = 'https://www.enterprisesg.gov.sg/-/media/esg/files/non-financial-assistance/for-companies/free-trade-agreements/uksfta/uksfta_text_9dec2020.pdf?la=en&hash=6AE336920662AC6A10E10B808DB588A2'

txt_1 = linkClicker(link_1)

link_2 = 'https://www.enterprisesg.gov.sg/-/media/esg/files/non-financial-assistance/for-companies/free-trade-agreements/uksfta/UKSFTA_Digital_Economy_Agreement.pdf'

txt_2 = linkClicker(link_2)

os.chdir('text')

try:
    os.chdir('ukfta')
except:
    os.mkdir('ukfta')
    os.chdir('ukfta')


bruteCleaner(txt_1, 'ukfta.txt')
bruteCleaner(txt_2, 'ukfta_dea.txt')
