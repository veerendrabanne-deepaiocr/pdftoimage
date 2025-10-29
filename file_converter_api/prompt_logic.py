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
        System Role: You are a global financial document expert specializing in data extraction schemas. Provide data in JSON format only.

        Task: For the country "{country_name}", return a detailed JSON object for building a data extraction prompt. Include country codes, currency, specific identifiers with their JSON keys and validation rules, detailed tax components, and any unique fields. Adhere strictly to the schema.

        Requested JSON Schema:
        {{
          "country_name": "string",
          "country_code_iso": "string (ISO 3166-1 alpha-2)",
          "currency_code": "string (ISO 4217)",
          "identifiers": [
            {{
              "name": "string (e.g., Goods and Services Tax ID)",
              "json_key": "string (e.g., gstin)",
              "validation_rule": "string (e.g., 15-char alphanumeric)",
              "type": "tax/company/banking",
              "common_on": ["invoice", "po", "quote"]
            }}
          ],
          "tax_components": [
            {{
              "name": "string (e.g., Central Goods and Services Tax)",
              "json_key": "string (e.g., cgst_amount)"
            }}
          ],
          "unique_fields": [
            {{
                "name": "string (e.g., Place of Supply)",
                "json_key": "string (e.g., place_of_supply)",
                "type": "string"
            }},
            {{
                "name": "string (e.g., Harmonized System of Nomenclature Code)",
                "json_key": "string (e.g., hsn_sac_code)",
                "scope": "line_item",
                "type": "string"
            }}
          ],
          "common_field_aliases": {{
            "total_amount": "string (Local term for Total)"
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



def build_dynamic_schema(country_data: dict, doc_type: str) -> dict:
    """
    Dynamically builds the JSON schema for a given document type based on country data.
    """
    doc_type_lower = doc_type.lower().replace(" ", "_").replace("rfq_", "")

    schema = {
        "document_type": doc_type_lower,
        "country_code": country_data.get("country_code_iso", ""),
    }

    # Add common document fields based on type
    if doc_type_lower == "invoice":
        schema.update({
            "invoice_number": "string",
            "invoice_date": "string(YYYY-MM-DD)",
            "due_date": "string(YYYY-MM-DD)",
        })
    elif doc_type_lower == "purchase_order":
        schema.update({
            "po_number": "string",
            "po_date": "string(YYYY-MM-DD)",
            "delivery_date": "string(YYYY-MM-DD)",
        })
    elif doc_type_lower == "quotation":
        schema.update({
            "quote_number": "string",
            "quote_date": "string(YYYY-MM-DD)",
            "valid_until": "string(YYYY-MM-DD)",
        })

    # Add unique root-level fields
    for field in country_data.get("unique_fields", []):
        if field.get("scope") != "line_item":
            schema[field["json_key"]] = field["type"]

    # Seller and Buyer details
    schema["seller_details"] = {"name": "string", "address": "string"}
    schema["buyer_details"] = {"name": "string", "address": "string"}

    for identifier in country_data.get("identifiers", []):
        if doc_type_lower in identifier.get("common_on", []):
            if identifier["type"] in ["tax", "company"]:
                schema["seller_details"][identifier["json_key"]] = "string"
                # Assume buyer might also have a tax/company ID
                if doc_type_lower != "purchase_order":
                    schema["buyer_details"][identifier["json_key"]] = "string"

    # Line Items
    line_item_schema = {
        "description": "string",
        "quantity": "number",
        "unit_price": "number",
        "total_amount": "number",
    }
    for field in country_data.get("unique_fields", []):
        if field.get("scope") == "line_item":
            line_item_schema[field["json_key"]] = field["type"]
    schema["line_items"] = [line_item_schema]

    # Financial Summary
    financial_summary = {"subtotal": "number"}
    for tax in country_data.get("tax_components", []):
        financial_summary[tax["json_key"]] = "number"

    total_amount_key = f"total_amount_{country_data.get('currency_code', '').lower()}"
    financial_summary[total_amount_key] = "number"
    schema["financial_summary"] = financial_summary

    # Payment Details
    payment_details = {}
    for identifier in country_data.get("identifiers", []):
        if doc_type_lower in identifier.get("common_on", []) and identifier["type"] == "banking":
            payment_details[identifier["json_key"]] = "string"
    if payment_details:
        schema["payment_details"] = payment_details

    return schema

def generate_extraction_prompt(country_data: dict, doc_type: str, model_name: str) -> str:
    """
    Generates the final extraction prompt for a given document type and AI model
    using the new, highly structured format.
    """
    # 1. Build the dynamic JSON schema
    schema_dict = build_dynamic_schema(country_data, doc_type)
    schema_json_br = json.dumps(schema_dict, indent=2).replace('\n', '<br>')

    # 2. Gather data for prompt sections
    country_name = country_data.get("country_name", "")
    currency = country_data.get("currency_code", "")
    identifiers = country_data.get("identifiers", [])
    tax_components = country_data.get("tax_components", [])

    # 3. Generate model-specific prompts
    if model_name == 'Gemini 2.5 Flash':
        # Build Document Context
        doc_context = f"The document originates from {country_name}. Expect {currency} currency, "
        if tax_components:
            doc_context += f"{tax_components[0]['name']} tax structures, "

        id_list = [f"{i['name']} ({i['json_key']})" for i in identifiers]
        doc_context += f"and specific identifiers: {', '.join(id_list)}."

        # Build Field-Specific Directives
        field_directives = "\n".join([f"* {i['json_key']}: {i['validation_rule']}." for i in identifiers])

        # Build tax logic directive
        if len(tax_components) > 1:
            tax_keys = [t['json_key'] for t in tax_components]
            field_directives += f"\n* Differentiate {', '.join(tax_keys)}. If one tax type applies, others should be 0 or null."

        # Assemble the final prompt
        prompt = (
            "System Role: You are a high-precision, autonomous data extraction engine (doc_parser_v4.2). "
            "Your sole function is to parse the provided document and output only a valid, minified JSON object "
            "adhering strictly to the schema.\n\n"
            f"Task: Extract key financial and identification data from the {country_name} {doc_type} provided.\n\n"
            f"Document Context: {doc_context}\n\n"
            "Core Execution Rules:\n"
            "1. Output MUST be JSON only: Your entire response must be a single, minified JSON object. No preamble, no markdown formatting, no explanations.\n"
            "2. Strict Schema Adherence: Follow the JSON Schema precisely. All specified keys must be present.\n"
            "3. Handle Missing Data: If a field's value is not found in the document, return null for that key. DO NOT omit the key.\n"
            "4. Normalization: Dates must be YYYY-MM-DD. Monetary values must be number type (float/integer), remove currency symbols.\n\n"
            f"Field-Specific Directives:\n{field_directives}\n\n"
            f"JSON Schema:\njson<br>{schema_json_br}<br>"
        )
        return prompt

    elif model_name == 'GPT 4.1':
        # Build Document Context
        id_details = ", ".join([f"{i['name']} ({i['validation_rule']})" for i in identifiers])
        tax_system_name = tax_components[0]['name'] if tax_components else "local"
        doc_context = (
            f"Source Document: {country_name} {doc_type}. Currency: {currency}. Tax System: {tax_system_name}. "
            f"Key Identifiers: {id_details}."
        )

        # Build Extraction Rules
        tax_logic_rule = ""
        if len(tax_components) > 1:
            tax_keys = [t['json_key'] for t in tax_components]
            tax_logic_rule = f"4. Tax Logic: Correctly identify and populate {', '.join(tax_keys)}. Non-applicable tax types should be 0 or null."

        extraction_rules = (
            "1. Strict Schema Adherence: Output JSON must match <output_schema>. All keys are mandatory.\n"
            "2. Null Handling: If data for a key is absent, its value must be null. Do not omit keys.\n"
            "3. Normalization: Dates=YYYY-MM-DD. Monetary=number (no symbols).\n"
            f"{tax_logic_rule}"
        )

        # Assemble the final prompt
        prompt = (
            "<system_prompt>\n"
            "You are doc_parser_ai_v4.2, optimized for high-fidelity structured data extraction. "
            "Your output must be a single, minified JSON object conforming exactly to the <output_schema>. "
            "Do not include explanations, markdown, or any non-JSON text.\n"
            "</system_prompt>\n\n"
            f"<document_context>\n{doc_context}\n</document_context>\n\n"
            f"<extraction_rules>\n{extraction_rules}\n</extraction_rules>\n\n"
            f"<output_schema>\njson<br>{schema_json_br}<br>\n</output_schema>"
        )
        return prompt

    return ""
