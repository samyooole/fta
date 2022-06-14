
import requests
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
