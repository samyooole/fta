from bs4 import BeautifulSoup
import requests
import re

# Access the ESG website with all the FTAs (website as of 23 May 2022)
"""
Please check if the ESG sidebar of FTAs is exhaustive
"""

url = "https://www.enterprisesg.gov.sg/non-financial-assistance/for-singapore-companies/free-trade-agreements/ftas/singapore-ftas"
page = requests.get(url)
soup = BeautifulSoup(page.text)
subsoup = soup.find_all(class_= "text-list")
esgweb = 'https://www.enterprisesg.gov.sg'

for fta in subsoup:
    """
    Iterate over the list of FTAs on the mainpage. Assumes that the list on ESG website is exhaustive.
    """
    fta_back_url = fta.find('a', href=True)['href']
    fta_url = esgweb + fta_back_url
    fta_page = requests.get(fta_url)
    fta_soup = BeautifulSoup(fta_page.text)
    mother_pdf_back_url = fta_soup.find(string=re.compile("Download")).parent['href']                 # Find link that downloads the FTA legal text
    
    """
    This is how ESG structures their FTA texts:
    - one mother PDF, which contains many children links to different PDFs
    - so we have to iterate over every link within each mother PDF to download the files
    """
    mother_pdf_url = esgweb + mother_pdf_back_url
    mother_pdf_page = requests.get(mother_pdf_url)
    mother_pdf_soup = BeautifulSoup(mother_pdf_page.text)