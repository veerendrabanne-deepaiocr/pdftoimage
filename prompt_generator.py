# -*- coding: utf-8 -*-
"""
This script dynamically generates country-specific AI prompts for data extraction
from financial documents and saves them to a CSV file.
"""

# Installation instructions:
# pip install google-generativeai python-dotenv

import os
import csv
import json
from copy import deepcopy
import google.generativeai as genai
from dotenv import load_dotenv

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

# --- Base Schema Templates ---

# Base JSON schema for Invoice extraction prompts
INVOICE_SCHEMA_TEMPLATE = {
    "document_type": "invoice",
    "country_code": "{COUNTRY_CODE_ISO}",
    "invoice_number": "string",
    "invoice_date": "string(YYYY-MM-DD)",
    "due_date": "string(YYYY-MM-DD)",
    "seller_details": {
        "name": "string",
        "address": "string",
        "{PRIMARY_TAX_ID_NAME}": "string",
        "{PRIMARY_COMPANY_ID_NAME}": "string",
    },
    "buyer_details": {
        "name": "string",
        "address": "string",
    },
    "line_items": [
        {
            "description": "string",
            "quantity": "number",
            "unit_price": "number",
            "total_price": "number"
        }
    ],
    "financial_summary": {
        "subtotal": "number",
        "{VAT_TAX_NAME}_amount": "number",
        "total_amount_{CURRENCY_CODE}": "number"
    },
    "payment_details": {
        "{PRIMARY_BANKING_DOMESTIC}": "string",
        "{PRIMARY_BANKING_INTERNATIONAL}": "string"
    }
}

# Base JSON schema for Purchase Order extraction prompts
PO_SCHEMA_TEMPLATE = {
    "document_type": "purchase_order",
    "country_code": "{COUNTRY_CODE_ISO}",
    "po_number": "string",
    "po_date": "string(YYYY-MM-DD)",
    "delivery_date": "string(YYYY-MM-DD)",
    "seller_details": {
        "name": "string",
        "address": "string",
    },
    "buyer_details": {
        "name": "string",
        "address": "string",
        "{PRIMARY_COMPANY_ID_NAME}": "string",
    },
    "line_items": [
        {
            "description": "string",
            "quantity": "number",
            "unit_price": "number",
            "total_price": "number"
        }
    ],
    "financial_summary": {
        "subtotal": "number",
        "total_amount_{CURRENCY_CODE}": "number"
    }
}

# Base JSON schema for RFQ Quotation extraction prompts
QUOTATION_SCHEMA_TEMPLATE = {
    "document_type": "quotation",
    "country_code": "{COUNTRY_CODE_ISO}",
    "quote_number": "string",
    "quote_date": "string(YYYY-MM-DD)",
    "valid_until": "string(YYYY-MM-DD)",
    "seller_details": {
        "name": "string",
        "address": "string",
        "{PRIMARY_TAX_ID_NAME}": "string",
    },
    "buyer_details": {
        "name": "string",
        "address": "string",
    },
    "line_items": [
        {
            "description": "string",
            "quantity": "number",
            "unit_price": "number",
            "total_price": "number"
        }
    ],
    "financial_summary": {
        "subtotal": "number",
        "total_amount_{CURRENCY_CODE}": "number"
    }
}


def fetch_country_data(country_name: str, api_key: str) -> dict | None:
    """
    Fetches structured data about financial document conventions for a given country.

    Args:
        country_name: The name of the country.
        api_key: The Gemini API key.

    Returns:
        A dictionary containing the country-specific data, or None if an error occurs.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        meta_prompt = f"""
        System Role: You are a global financial document expert. Provide data in JSON format only.

        Task: For the country "{country_name}", return a JSON object containing typical
        identifiers, banking details, currency, and common field names found on business
        Invoices, Purchase Orders, and Quotations. Adhere strictly to the requested JSON schema.
        If specific information is unavailable or not applicable for a field, use `null` or an
        empty list [] as appropriate per the schema.

        Requested JSON Schema:
        {{
            "country_name": "string",
            "currency_code": "string (ISO 4217)",
            "tax_ids": [
                {{"name": "string (Local name, e.g., VAT, GST)", "format_description": "string", "common_on_invoice": "boolean", "common_on_po": "boolean", "common_on_quote": "boolean"}}
            ],
            "company_ids": [
                {{"name": "string (Local name, e.g., CRN, EIN)", "format_description": "string", "common_on_invoice": "boolean", "common_on_po": "boolean", "common_on_quote": "boolean"}}
            ],
            "banking_details": [
                {{"name": "string (Local name, e.g., IBAN, SWIFT, Routing Number)", "format_description": "string", "type": "domestic/international/local_system"}}
            ],
            "common_field_aliases": {{
                "invoice": "string (Local term for Invoice)",
                "purchase_order": "string (Local term for PO)",
                "quotation": "string (Local term for Quote)",
                "due_date": "string (Local term for Due Date)",
                "total_amount": "string (Local term for Total)",
                "vat_tax": "string (Local term for VAT/Sales Tax)"
            }}
        }}
        """

        response = model.generate_content(
            meta_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"An error occurred while fetching data for {country_name}: {e}")
        return None


def generate_extraction_prompt(country_data: dict, doc_type: str, model_name: str) -> str:
    """
    Generates the final extraction prompt for a given document type and AI model.

    Args:
        country_data: The dictionary of country-specific data.
        doc_type: The type of document (Invoice, Purchase Order, RFQ Quotation).
        model_name: The name of the target AI model.

    Returns:
        The complete, formatted prompt string.
    """
    if doc_type == 'Invoice':
        schema_template = deepcopy(INVOICE_SCHEMA_TEMPLATE)
    elif doc_type == 'Purchase Order':
        schema_template = deepcopy(PO_SCHEMA_TEMPLATE)
    else:
        schema_template = deepcopy(QUOTATION_SCHEMA_TEMPLATE)

    # Populate the schema with country-specific data
    schema_template['country_code'] = country_data.get('country_name', '')

    # Financial summary
    if 'financial_summary' in schema_template:
        fin_summary = schema_template['financial_summary']
        new_fin_summary = {}
        for key, value in fin_summary.items():
            new_key = key.replace('{CURRENCY_CODE}', country_data.get('currency_code', ''))
            new_key = new_key.replace('{VAT_TAX_NAME}', country_data.get('common_field_aliases', {}).get('vat_tax', 'vat_tax'))
            new_fin_summary[new_key] = value
        schema_template['financial_summary'] = new_fin_summary

    # Logic to determine the correct key for commonality based on doc_type
    if doc_type == 'Invoice':
        common_tax_key = 'common_on_invoice'
        common_company_key = 'common_on_invoice'
    elif doc_type == 'Purchase Order':
        common_tax_key = 'common_on_po'
        common_company_key = 'common_on_po'
    else: # RFQ Quotation
        common_tax_key = 'common_on_quote'
        common_company_key = 'common_on_quote'

    # Details sections (seller and buyer)
    for detail_section in ['seller_details', 'buyer_details']:
        if detail_section in schema_template:
            details = schema_template[detail_section]
            new_details = {}
            for key, value in details.items():
                new_key = key.replace('{PRIMARY_TAX_ID_NAME}', next((tid['name'] for tid in country_data.get('tax_ids', []) if tid.get(common_tax_key)), 'tax_id'))
                new_key = new_key.replace('{PRIMARY_COMPANY_ID_NAME}', next((cid['name'] for cid in country_data.get('company_ids', []) if cid.get(common_company_key)), 'company_id'))
                new_details[new_key] = value
            schema_template[detail_section] = new_details

    # Payment details
    if 'payment_details' in schema_template:
        payment_details = schema_template['payment_details']
        new_payment_details = {}
        for key, value in payment_details.items():
            new_key = key.replace('{PRIMARY_BANKING_DOMESTIC}', next((bd['name'] for bd in country_data.get('banking_details', []) if bd.get('type') == 'domestic'), 'domestic_bank_account'))
            new_key = new_key.replace('{PRIMARY_BANKING_INTERNATIONAL}', next((bd['name'] for bd in country_data.get('banking_details', []) if bd.get('type') == 'international'), 'international_bank_account'))
            new_payment_details[new_key] = value
        schema_template['payment_details'] = new_payment_details

    # Convert the populated schema to a formatted JSON string
    schema_json = json.dumps(schema_template, indent=2)

    # Construct the prompt text
    prompt_text = f"""
System Role: doc_parser_v4.2
Task: Extract data for the {doc_type}.
Document Context:
- Country: {country_data.get('country_name', '')}
- Currency: {country_data.get('currency_code', '')}
- Key Identifiers: {', '.join([tid['name'] for tid in country_data.get('tax_ids', [])])}
- Banking Details: {', '.join([bd['name'] for bd in country_data.get('banking_details', [])])}

Core Execution Rules:
- Output JSON ONLY.
- Adhere strictly to the schema.
- All keys are MANDATORY.
- Use null for missing values.
- Normalize dates to YYYY-MM-DD.
- Normalize monetary values to numbers (strip symbols).
"""

    # Apply model-specific formatting
    if model_name == 'Gemini 2.5 Flash':
        return f"**System Role:**\n{prompt_text}\n**JSON Schema:**\n```json\n{schema_json}\n```"
    elif model_name == 'GPT 4.1':
        # Improved formatting for GPT 4.1 with structured XML
        country_context = f"- Country: {country_data.get('country_name', '')}\n- Currency: {country_data.get('currency_code', '')}"
        key_identifiers = f"- Key Identifiers: {', '.join([tid['name'] for tid in country_data.get('tax_ids', [])])}"
        banking_details = f"- Banking Details: {', '.join([bd['name'] for bd in country_data.get('banking_details', [])])}"

        system_prompt_content = "System Role: doc_parser_v4.2\nTask: Extract data for the {doc_type}."
        extraction_rules_content = "Core Execution Rules:\n- Output JSON ONLY.\n- Adhere strictly to the schema.\n- All keys are MANDATORY.\n- Use null for missing values.\n- Normalize dates to YYYY-MM-DD.\n- Normalize monetary values to numbers (strip symbols)."

        return (
            f"<system_prompt>{system_prompt_content}</system_prompt>\n"
            f"<document_context>\n{country_context}\n{key_identifiers}\n{banking_details}\n</document_context>\n"
            f"<extraction_rules>\n{extraction_rules_content}\n</extraction_rules>\n"
            f"<output_schema>json\n{schema_json}\n</output_schema>"
        )

    return ""


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
            country_data = fetch_country_data(country, GEMINI_API_KEY)

            if country_data:
                for doc_type in DOCUMENT_TYPES:
                    for model_name in TARGET_MODELS:
                        prompt = generate_extraction_prompt(country_data, doc_type, model_name)
                        csv_writer.writerow([country, doc_type, model_name, prompt])
            else:
                print(f"Warning: Failed to fetch data for {country}. Skipping.")

    print(f"\nScript finished. Prompts saved to {OUTPUT_CSV_FILENAME}")


if __name__ == "__main__":
    main()
