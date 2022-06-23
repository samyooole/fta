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
            text=page.extract_text()
            plaintext+=text

    os.remove('temp3.pdf')
    return plaintext


acia = linkClicker("http://investasean.asean.org/files/upload/Doc%2005%20-%20ACIA.pdf")

with open("acia.txt", "w", encoding="utf-8") as outfile:
            outfile.write(acia)

acfta_invest = linkClicker("http://fta.mofcom.gov.cn/inforimages/200908/20090817113007764.pdf")

plaintext=plaintext.replace('PDF 文件使用 "pdfFactory Pro" 试用版本创建 www.fineprint.cn', "")

with open("acfta_invest.txt", "w", encoding="utf-8") as outfile:
            outfile.write(plaintext)