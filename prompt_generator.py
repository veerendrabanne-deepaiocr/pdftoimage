# -*- coding: utf-8 -*-
"""
This script dynamically generates country-specific AI prompts for data extraction
from financial documents and saves them to a CSV file.
"""

# Installation instructions:
# pip install google-generativeai python-dotenv

import os
import csv
from dotenv import load_dotenv
from file_converter_api import prompt_logic

# --- Setup & Configuration ---

# Load environment variables from .env file
load_dotenv()

# Securely fetch the Gemini API key from an environment variable
# To set this up, create a .env file in the same directory as this script
# and add the following line to it:
# GEMINI_API_KEY="your_api_key"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global lists and variables for easy modification
TARGET_COUNTRIES = [
    'USA', 'United Kingdom', 'Canada', 'Germany', 'France', 'Japan', 'China',
    'India', 'Ukraine', 'Australia', 'Brazil', 'Sweden'
]
DOCUMENT_TYPES = ['Invoice', 'Purchase Order', 'RFQ Quotation']
TARGET_MODELS = ['Gemini 2.5 Flash', 'GPT 4.1']
OUTPUT_CSV_FILENAME = 'generated_prompts.csv'


def main():
    """
    Main function to generate AI prompts and save them to a CSV file.
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    with open(OUTPUT_CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Country', 'Document Type', 'Model', 'Prompt'])

        for country in TARGET_COUNTRIES:
            print(f"Processing country: {country}...")
            country_data = prompt_logic.fetch_country_data(country, GEMINI_API_KEY)

            if country_data:
                for doc_type in DOCUMENT_TYPES:
                    for model_name in TARGET_MODELS:
                        prompt = prompt_logic.generate_extraction_prompt(country_data, doc_type, model_name)
                        csv_writer.writerow([country, doc_type, model_name, prompt])
            else:
                print(f"Warning: Failed to fetch data for {country}. Skipping.")

    print(f"\nScript finished. Prompts saved to {OUTPUT_CSV_FILENAME}")


if __name__ == "__main__":
    main()
