from bs4 import BeautifulSoup
import requests
import re
import pdfx
import fitz
import os

# Define linkClicker():
"""
- inputs: an online pdf link
- outputs: string
"""

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



# Access the ESG website with all the FTAs (website as of 23 May 2022)
"""
Please check if the ESG sidebar of FTAs is exhaustive
"""

url = "https://www.enterprisesg.gov.sg/non-financial-assistance/for-singapore-companies/free-trade-agreements/ftas/singapore-ftas"
page = requests.get(url)
soup = BeautifulSoup(page.text)
subsoup = soup.find_all(class_= "text-list")
subsoup = subsoup[0]

subsoup = subsoup.find_all('a', href=True)

esgweb = 'https://www.enterprisesg.gov.sg'

for fta in subsoup:
    """
    Iterate over the list of FTAs on the mainpage. Assumes that the list on ESG website is exhaustive.
    """
    fta_back_url = fta['href']
    fta_url = esgweb + fta_back_url
    fta_page = requests.get(fta_url)
    fta_soup = BeautifulSoup(fta_page.text)

    try:
        mother_pdf_back_url = fta_soup.find(string=re.compile("Download")).parent['href']                 # Find link that downloads the FTA legal text
    except:
        continue # non-conformers: ukfta
    """
    This is how ESG structures their FTA texts:
    - one mother PDF, which contains many children links to different PDFs
    - so we have to iterate over every link within each mother PDF to download the files

    - NOTE: we are currently ignoring scanned texts. should not presently be a problem because those are mostly memos instead of actual agreements
    """
    
    if "http" in mother_pdf_back_url:
        mother_pdf_url = mother_pdf_back_url
    else:
        mother_pdf_url = esgweb + mother_pdf_back_url
    
    try:

        mother_pdf_page = requests.get(mother_pdf_url)

    except:
        mother_pdf_url = esgweb + "/" + mother_pdf_back_url
        mother_pdf_page = requests.get(mother_pdf_url)

    f = open("temp.pdf", "wb")                                              # I don't quite like this move - could be more elegant
    f.write(mother_pdf_page.content)                                        # We write it off as a pdf then read it again, because pypdf2 does not (?) have an direct parser. To find and improve.
    f.close()

    pdf = pdfx.PDFx('temp.pdf')

    try:
        sub_ftas = pdf.get_references_as_dict()['pdf']
    except:
        sub_ftas = pdf.get_references_as_dict()['url']


    """
    - identify which fta we want to get the text for
    - create a folder for that text
    - identify each section of the fta
    - dump .txt files for each section into the aforementioned folder, with a descriptive name
    """
    prefix = '/non-financial-assistance/for-singapore-companies/free-trade-agreements/ftas/singapore-ftas/'
    fta_name = re.sub(pattern = prefix, repl = '', string = fta_back_url, flags = re.IGNORECASE)
    os.chdir('text') # will probably throw an error on loop
    try:
        os.makedirs(fta_name)
    except:
        print("dir alr made")
    os.chdir(fta_name)

    for sub_fta in sub_ftas:
        
        if re.search('terms-of-use', sub_fta.lower()) is not None :
            continue # I don't want to go to the terms of use website
        
        sub_fta_text = linkClicker(sub_fta)
        #simple cleaning: NOTE should try to be less brute force? very manual rn based on observations
        
        try:
            sub_fta_text = str.replace(sub_fta_text, "\n", "")
        except:
            continue
        sub_fta_text = str.replace(sub_fta_text, "™", "'")
        #finding a name
        sub_fta_name = re.sub(pattern = fta_name, repl = '', string = sub_fta, flags = re.IGNORECASE)
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
    
    # write the mother file in case it is in fact the full text

    f = open("mothertext.pdf", "wb")                                              # I don't quite like this move - could be more elegant
    f.write(mother_pdf_page.content)                                        # We write it off as a pdf then read it again, because pypdf2 does not (?) have an direct parser. To find and improve.
    f.close()


    os.chdir(os.path.dirname(os.getcwd())) # don't you just love convoluted russian doll functions
    os.chdir(os.path.dirname(os.getcwd()))

