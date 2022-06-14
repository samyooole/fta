import requests
import re
import fitz
import os
import pdfplumber as pb


def linkClicker(link):
    page = requests.get(link)
    f = open("temp3.pdf", "wb")                                              # I don't quite like this move - could be more elegant
    f.write(page.content)                                        # We write it off as a pdf then read it again, because pypdf2 does not (?) have an direct parser. To find and improve.
    f.close()

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

    os.remove('temp3.pdf')
    return plaintext


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
        sub_fta = item['uri']

        if re.search('terms-of-use', sub_fta.lower()) is not None :
            continue # I don't want to go to the terms of use website

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

        if sub_fta_text =='':
            continue
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
        with open(sub_fta_name + ".txt", "w", encoding="utf-8") as outfile:
            outfile.write(sub_fta_text)

        print(sub_fta_name + " from " + fta_name + " is done")
        
