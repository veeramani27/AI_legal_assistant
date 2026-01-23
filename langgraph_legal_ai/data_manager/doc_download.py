import os

import requests
import urllib3
from bs4 import BeautifulSoup

# Disable SSL warnings if the gov site has certificate issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from target_laws import TARGET_LAWS


def download_law_pdfs(laws_dict, download_folder="legal_corpus"):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Mimic a real browser to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for law_name, url in laws_dict.items():
        print(f"Scraping {law_name}...")
        try:
            # 1. Get the 'Handle' page
            response = requests.get(url, headers=headers, verify=False)
            soup = BeautifulSoup(response.content, "html.parser")

            pdf_link = None
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/bitstream/" in href and href.lower().endswith(".pdf"):
                    pdf_link = "https://www.indiacode.nic.in" + href
                    break

            if pdf_link:
                print(f"Found PDF: {pdf_link}")
                pdf_response = requests.get(
                    pdf_link, headers=headers, verify=False, stream=True
                )

                file_path = os.path.join(download_folder, f"{law_name}.pdf")
                with open(file_path, "wb") as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully saved {law_name}.pdf")
            else:
                print(
                    f"Could not find PDF link for {law_name}. The site structure might have changed."
                )

        except Exception as e:
            print(f"Error downloading {law_name}: {e}")


download_law_pdfs(TARGET_LAWS)
