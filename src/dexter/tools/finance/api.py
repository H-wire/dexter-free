import os
import requests

####################################
# API Configuration
####################################

financial_datasets_api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
financial_datasets_base_url = os.getenv(
    "FINANCIAL_DATASETS_BASE_URL",
    "https://api.financialdatasets.ai",
).rstrip("/")


def call_api(endpoint: str, params: dict) -> dict:
    """Helper function to call the Financial Datasets API."""
    base = financial_datasets_base_url or "https://api.financialdatasets.ai"
    url = f"{base}{endpoint}"
    headers = {"x-api-key": financial_datasets_api_key}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()
