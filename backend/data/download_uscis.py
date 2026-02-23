"""
USCIS form instructions are public domain — US government documents.
This script downloads the 25 most common form instructions.
"""
import httpx
import os
from pathlib import Path

# All public domain — no copyright issues
USCIS_FORMS = {
    # Adjustment of status / green card
    "I-485": "https://www.uscis.gov/sites/default/files/document/forms/i-485instr.pdf",
    "I-130": "https://www.uscis.gov/sites/default/files/document/forms/i-130instr.pdf",
    "I-130A": "https://www.uscis.gov/sites/default/files/document/forms/i-130ainstr.pdf",
    "I-864": "https://www.uscis.gov/sites/default/files/document/forms/i-864instr.pdf",
    "I-864A": "https://www.uscis.gov/sites/default/files/document/forms/i-864ainstr.pdf",
    "I-765": "https://www.uscis.gov/sites/default/files/document/forms/i-765instr.pdf",
    "I-131": "https://www.uscis.gov/sites/default/files/document/forms/i-131instr.pdf",
    "I-693": "https://www.uscis.gov/sites/default/files/document/forms/i-693instr.pdf",

     # === STUDENT VISAS (F1, M1, J1) ===
    "I-20": "https://studyinthestates.dhs.gov/sites/default/files/2023-04/I-20_Instructions.pdf",  # SEVIS form (DSO issues)
    "I-539": "https://www.uscis.gov/sites/default/files/document/forms/i-539instr.pdf",  # F1 extension/change
    "I-765": "https://www.uscis.gov/sites/default/files/document/forms/i-765instr.pdf",  # OPT/STEM OPT EAD
    "I-983": "https://www.ice.gov/doclib/sevis/pdf/i-983-instr.pdf",  # STEM OPT Training Plan
    "I-131": "https://www.uscis.gov/sites/default/files/document/forms/i-131instr.pdf",  # Travel/Advance Parole
    
    # === WORK TRANSITIONS (F1 → H1B/O1) ===
    "I-129": "https://www.uscis.gov/sites/default/files/document/forms/i-129instr.pdf",  # H1B/L1/O1 petition
    "I-907": "https://www.uscis.gov/sites/default/files/document/forms/i-907instr.pdf",  # Premium processing

    # Employment-based, nonimmigrant workers
    "I-140": "https://www.uscis.gov/sites/default/files/document/forms/i-140instr.pdf",
    "I-129": "https://www.uscis.gov/sites/default/files/document/forms/i-129instr.pdf",
    "I-539": "https://www.uscis.gov/sites/default/files/document/forms/i-539instr.pdf",

    # Green card maintenance / replacement
    "I-90": "https://www.uscis.gov/sites/default/files/document/forms/i-90instr.pdf",
    "I-751": "https://www.uscis.gov/sites/default/files/document/forms/i-751instr.pdf",
    "I-829": "https://www.uscis.gov/sites/default/files/document/forms/i-829instr.pdf",

    # Naturalization / citizenship
    "N-400": "https://www.uscis.gov/sites/default/files/document/forms/n-400instr.pdf",
    "N-600": "https://www.uscis.gov/sites/default/files/document/forms/n-600instr.pdf",
    "N-600K": "https://www.uscis.gov/sites/default/files/document/forms/n-600kinstr.pdf",

    # Humanitarian / special immigrant
    "I-589": "https://www.uscis.gov/sites/default/files/document/forms/i-589instr.pdf",
    "I-360": "https://www.uscis.gov/sites/default/files/document/forms/i-360instr.pdf",
    "I-730": "https://www.uscis.gov/sites/default/files/document/forms/i-730instr.pdf",

    # Fiancé(e) / family-related extra
    "I-129F": "https://www.uscis.gov/sites/default/files/document/forms/i-129finstr.pdf",
    "I-134": "https://www.uscis.gov/sites/default/files/document/forms/i-134instr.pdf",

    # Investor
    "I-526": "https://www.uscis.gov/sites/default/files/document/forms/i-526instr.pdf",
    "I-526E": "https://www.uscis.gov/sites/default/files/document/forms/i-526einstr.pdf",

    # Misc helpful
    "I-824": "https://www.uscis.gov/sites/default/files/document/forms/i-824instr.pdf",
}

def download_forms(output_dir: str = "data/uscis_pdfs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for form_number, url in USCIS_FORMS.items():
        output_path = f"{output_dir}/{form_number}_instructions.pdf"

        if os.path.exists(output_path):
            print(f"{output_path} already exists, skipping download.")
            continue

        try:
            print(f" Dowloading {form_number} instructions from {url}...")
            response = httpx.get(url, follow_redirects=True, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes

            with open(output_path, "wb") as f: # Save the PDF content to a file
                f.write(response.content)
            print(f" {form_number}: saves ({len(response.content) // 1024} KB)")
        
        except Exception as e:
            print(f" {form_number}: Failed to download from {url}. Error: {str(e)}")

if __name__ == "__main__":
    download_forms()
