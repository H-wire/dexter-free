import requests

from dexter.config import FINANCIAL_DATASETS_API_KEY

####################################
# API Configuration
####################################


def _get_api_key() -> str:
    """Return the configured API key or raise if missing."""
    api_key = FINANCIAL_DATASETS_API_KEY
    if not api_key:
        raise EnvironmentError(
            "FINANCIAL_DATASETS_API_KEY is required to call FinancialDatasets tools. "
            "Set it in your environment or .env (copy from env.example)."
        )
    return api_key


def call_api(endpoint: str, params: dict) -> dict:
    """Helper function to call the Financial Datasets API."""
    base_url = "https://api.financialdatasets.ai"
    url = f"{base_url}{endpoint}"
    headers = {"x-api-key": _get_api_key()}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [401, 403]:
            raise EnvironmentError(
                "Invalid FINANCIAL_DATASETS_API_KEY. Please check your API key."
            )
        elif e.response.status_code == 429:
            raise ConnectionError(
                "Rate limit exceeded. Please wait and try again."
            )
        else:
            raise
