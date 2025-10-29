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
    'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
    'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi',
    'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Congo-Brazzaville)', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czechia (Czech Republic)',
    'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
    'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini (fmr. "Swaziland")', 'Ethiopia',
    'Fiji', 'Finland', 'France',
    'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana',
    'Haiti', 'Holy See', 'Honduras', 'Hungary',
    'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
    'Jamaica', 'Japan', 'Jordan',
    'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan',
    'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
    'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar (formerly Burma)',
    'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway',
    'Oman',
    'Pakistan', 'Palau', 'Palestine State', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
    'Qatar',
    'Romania', 'Russia', 'Rwanda',
    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria',
    'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu',
    'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States of America', 'Uruguay', 'Uzbekistan',
    'Vanuatu', 'Venezuela', 'Vietnam',
    'Yemen',
    'Zambia', 'Zimbabwe'
]
DOCUMENT_TYPES = ['Invoice', 'Purchase Order', 'RFQ Quotation', 'Travel Itinerary', 'Expense Report']
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
