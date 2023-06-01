from bs4 import BeautifulSoup
import requests
import re
import pdfx
import fitz
import os
import pdfplumber as pb
import camelot as cm
import json

# Define linkClicker():
"""
- inputs: an online pdf link
- outputs: string
"""

def linkClicker(link):
    page = requests.get(link)
    f = open("temp3.pdf", "wb")                                              # I don't quite like this move - could be more elegant
    f.write(page.content)                                        # We write it off as a pdf then read it again, because pypdf2 does not (?) have an direct parser. To find and improve.
    f.close()
    try:
        tables = cm.read_pdf('temp3.pdf', pages = '1-end')
    except:
        print('tables are wonky')
        tables=[]
    
    if len(tables)!=0:
        """
    first, go through the tables and get them in raw form (pandas)
    """
        lot = []
        for idx, table in enumerate(tables):
            newtable = table.df

            # do some newline cleaning and whitespace stripping

            def clean_newlines(str):
                return str.replace("\n", " ")

            def strip_whitespaces(str):
                mid = str.strip()
                return " ".join(mid.split())

            newtable = newtable.applymap(clean_newlines)
            newtable = newtable.applymap(strip_whitespaces)

            newtable.columns = newtable.iloc[0] # make first row as header
            newtable = newtable.drop([0])
            
            lot.append(newtable) # for now, we go with a naive approach that every visually separate table is in fact a separate table
            #backward check and append to previous df if necessary
            """
            if idx > 0:
                if newtable.columns.tolist() == lot[-1].columns.tolist():
                    appendedtable = lot[-1].append(newtable)
                    lot[-1] = appendedtable
                else:
                    lot.append(newtable)
            else:
                lot.append(newtable)
            """
            lod = []
            for table in lot:
                jsontable = table.to_dict()
                lod.append(jsontable)
    else:
        lod=[]

    

    plaintext=""
    with pb.open("temp3.pdf") as pdf:
        for page in pdf.pages:
            # Get the bounding boxes of the tables on the page.
            if (page.curves == []) & (page.edges ==[]):
                text=page.extract_text()
                plaintext+=text
            else:
                bboxes = [
                    table.bbox
                    for table in page.find_tables(
                        table_settings={
                            "vertical_strategy": "explicit",
                            "horizontal_strategy": "explicit",
                            "explicit_vertical_lines": page.curves + page.edges,
                            "explicit_horizontal_lines": page.curves + page.edges,
                        }
                    )
                ]

                def not_within_bboxes(obj):
                    """
                    credit: samkit jain
                    Check if the object is in any of the table's bbox."""
                    def obj_in_bbox(_bbox):
                        """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
                        v_mid = (obj["top"] + obj["bottom"]) / 2
                        h_mid = (obj["x0"] + obj["x1"]) / 2
                        x0, top, x1, bottom = _bbox
                        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)

                    return not any(obj_in_bbox(__bbox) for __bbox in bboxes)
                text = page.filter(not_within_bboxes).extract_text()
                plaintext += text

    
    newdict=    {
            'plaintext':plaintext,
            'tables':lod,
        }

    os.remove('temp3.pdf')
    return newdict

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


    def pdfReturner(pdf):
        try: 
            returnee = pdf.get_references_as_dict()['pdf']
            return returnee
        except:
            empty=[]
            return empty

    def urlReturner(pdf):
        try: 
            returnee = pdf.get_references_as_dict()['url']
            return returnee
        except:
            empty=[]
            return empty

    sub_ftas = pdfReturner(pdf) + urlReturner(pdf)
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

        if sub_fta.endswith("free-trade-agreements/ASEAN-China-FTA/Legal-Text"):
            continue

        if 'enterprisesg' not in sub_fta:
            continue

        if "static_post" in sub_fta:
            continue

        if 'error' in requests.get(sub_fta).url:
            continue
        
        try:
            sub_fta_text = linkClicker(sub_fta)
        except:
            continue
        if (sub_fta_text['plaintext'] == '') & (sub_fta_text['tables'] == []):
            continue
        
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
        with open(sub_fta_name + ".json", "w") as outfile:
            json.dump(sub_fta_text, outfile)

        print(sub_fta_name + " from " + fta_name + " is done")
    # write the mother file in case it is in fact the full text

    f = open("mothertext.pdf", "wb")                                              # I don't quite like this move - could be more elegant
    f.write(mother_pdf_page.content)                                        # We write it off as a pdf then read it again, because pypdf2 does not (?) have an direct parser. To find and improve.
    f.close()


    os.chdir(os.path.dirname(os.getcwd())) # don't you just love convoluted russian doll functions
    os.chdir(os.path.dirname(os.getcwd()))

