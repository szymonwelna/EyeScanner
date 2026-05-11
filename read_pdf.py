import pdfplumber

def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_text = read_pdf("DnoOka.pdf")
    print(pdf_text)