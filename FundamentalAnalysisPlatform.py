import streamlit as st
import pandas as pd
import requests  # For making API calls to FMP
import plotly.express as px
import google.generativeai as genai
import traceback
import os
import re
from datetime import datetime, timedelta
import numpy as np  # For checking numeric types and isnan

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Financial Analysis Toolkit")

# --- API Key Loading ---
# Try to load .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.sidebar.info("Consider installing `python-dotenv` and creating a `.env` file for local API key management.")

FMP_API_KEY = os.environ.get("FMP_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # For Gemini
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")  # For fallback

if not FMP_API_KEY:
    st.error("FMP_API_KEY environment variable not found. Please set it (e.g., in a .env file or your system environment) to use this application.")
    st.stop()
# GOOGLE_API_KEY is checked within its function
# ALPHA_VANTAGE_API_KEY will be checked before use in fetch function


FMP_BASE_URL = "https://financialmodelingprep.com/api"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


# --- Load Ticker List from FMP ---
@st.cache_data(ttl=24 * 3600)  # Cache the stock list for 24 hours
def load_ticker_file():
    """
    Loads ticker symbols and names from the FMP API.
    Filters for major U.S. exchanges like NASDAQ and NYSE.
    Returns a DataFrame with 'Symbol' and 'Display Label'.
    """
    fmp_stock_list_url = f"{FMP_BASE_URL}/v3/stock/list?apikey={FMP_API_KEY}"
    
    data_for_df = []
    major_exchanges = ["NASDAQ", "NYSE", "AMEX"] 

    try:
        response = requests.get(fmp_stock_list_url, timeout=20) 
        response.raise_for_status()
        stock_list_data = response.json()

        if not stock_list_data or not isinstance(stock_list_data, list):
            st.sidebar.error("Could not fetch or parse stock list from FMP. Response was not a valid list.")
            return pd.DataFrame(columns=['Symbol', 'Display Label'])

        for stock_item in stock_list_data:
            symbol = stock_item.get('symbol')
            name = stock_item.get('name')
            exchange = stock_item.get('exchangeShortName') 
            stock_type = stock_item.get('type')

            if symbol and name and exchange and stock_type == 'stock':
                if exchange in major_exchanges:
                    display_label = f"{symbol} - {name}"  # Removed exchange from display label
                    data_for_df.append({'Symbol': symbol, 'Display Label': display_label})
        
        if not data_for_df:
            st.sidebar.warning("No stocks found matching filter criteria from FMP stock list.")
            return pd.DataFrame(columns=['Symbol', 'Display Label'])

        df = pd.DataFrame(data_for_df)
        return df.sort_values(by="Display Label").reset_index(drop=True)

    except requests.exceptions.HTTPError as http_err:
        st.sidebar.error(f"FMP API HTTP Error fetching stock list: {http_err}")
        if http_err.response:
             st.sidebar.error(f"Response content: {http_err.response.text[:200]}")
        return pd.DataFrame(columns=['Symbol', 'Display Label'])
    except requests.exceptions.RequestException as req_err:
        st.sidebar.error(f"FMP Request Error fetching stock list: {req_err}")
        return pd.DataFrame(columns=['Symbol', 'Display Label'])
    except Exception as e:
        st.sidebar.error(f"Error processing stock list from FMP: {e}")
        st.sidebar.error(traceback.format_exc())
        return pd.DataFrame(columns=['Symbol', 'Display Label'])

# --- Helper Functions ---
def snake_case_to_title(snake_case_str):
    """Converts a camelCase string to a Title Case string."""
    if not isinstance(snake_case_str, str):
        return snake_case_str
    # Add a space before each capital letter, but not at the beginning of the string
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', snake_case_str)
    # Capitalize the first letter of each word
    return s.title()

def convert_to_shorthand(number):
    """Convert large numbers to K/M/B format"""
    abs_num = abs(number)
    sign = '-' if number < 0 else ''
    if abs_num >= 1_000_000_000:
        return f"{sign}{abs_num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{sign}{abs_num / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{sign}{abs_num / 1_000:.2f}K"
    return f"{sign}{abs_num:.2f}"

def format_value(value, format_type="currency", na_rep="N/A"):
    """Formats numbers for display with comma separators."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return na_rep
    try:
        val_float = float(value)
        if format_type == "currency":
            # Format as integer if whole number, else two decimals with commas
            if val_float.is_integer():
                return f"${val_float:,.0f}"
            else:
                return f"${val_float:,.2f}"
        if format_type == "currency_precise":
            return f"${val_float:,.2f}"  # Always show two decimals
        if format_type == "ratio":
            return f"{val_float:.2f}"  # Keep ratio formatting as is
        if format_type == "percent":
            return f"{val_float:.2%}"  # Percentage formatting
        if format_type == "number":
            # Format counts with commas and no decimals
            return f"{val_float:,.0f}"
    except (ValueError, TypeError):
        return str(value)
    return str(value)

def format_statement_values(df):
    """
    This is the definitive function to format a DataFrame for display.
    It pre-formats numbers into strings, bypassing .style object issues.
    It inspects the row index to apply context-specific formatting (e.g., %, $, x)
    and provides a readable, comma-rich format.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # Create a new DataFrame of object type to store string-formatted numbers
    df_display = df.copy().astype(object)

    for idx in df_display.index:
        for col in df_display.columns:
            value = df_display.loc[idx, col]

            # Check if value is a valid number (integer or float)
            if not pd.api.types.is_number(value) or pd.isna(value):
                df_display.loc[idx, col] = 'N/A'  # Represent non-numbers or NaNs consistently
                continue

            val_float = float(value)
            idx_lower = str(idx).lower() # Use lowercased index for matching

            # --- Apply formatting rules based on keywords in the row index ---

            # Rule 1: Format 'Cik' and 'Calendar Year' as whole numbers without commas
            if idx_lower in ['cik', 'calendar year', 'year']:
                formatted_value = f"{val_float:.0f}"

            # Rule 2: Percentages (Margins, Rates, Returns)
            elif any(term in idx_lower for term in ['margin', 'rate', 'yield', 'roe', 'roa', 'payout']):
                formatted_value = f"{val_float:.2%}"

            # Rule 3: Ratios and Multipliers (usually end with 'x')
            elif any(term in idx_lower for term in ['ratio', 'turnover', 'leverage', 'coverage', 'multiplier']) and 'days' not in idx_lower:
                formatted_value = f"{val_float:,.2f}x"

            # Rule 4: Per-Share values (typically currency)
            elif 'eps' in idx_lower:
                formatted_value = f"${val_float:,.2f}"

            # Rule 5: Default for all other numeric values (e.g., revenue, assets)
            else:
                # For numbers that are whole, format without decimal places.
                if val_float == int(val_float):
                    formatted_value = f"{val_float:,.0f}"
                # Otherwise, use two decimal places.
                else:
                    formatted_value = f"{val_float:,.2f}"

            # Assign the formatted string back to the display DataFrame
            df_display.loc[idx, col] = formatted_value

    return df_display


def get_safe_value(data_structure, keys, default=None):
    """
    Safely get a value from a Pandas Series or a dictionary.
    `keys` can be a single key string or a list of potential FMP field names to try.
    """
    if not isinstance(keys, list):
        keys = [keys]

    if isinstance(data_structure, pd.Series):
        for key in keys:
            if key in data_structure.index and pd.notna(data_structure[key]) and data_structure[key] != "":
                return data_structure[key]
    elif isinstance(data_structure, dict): 
        for key in keys:
            if data_structure and key in data_structure and data_structure[key] is not None and data_structure[key] != "":
                return data_structure[key]
    return default

def get_normalized_fcf(ratios_df, periods=3):
    """Computes the average FCF over the N most recent periods."""
    if ratios_df is None or ratios_df.empty or 'Free Cash Flow (FCF)' not in ratios_df.index:
        return None
    fcf_series = ratios_df.loc['Free Cash Flow (FCF)'].dropna()
    if fcf_series.empty:
        return None
    fcf_values = fcf_series.iloc[:periods].astype(float)
    return fcf_values.mean() if not fcf_values.empty else None

# --- AI Summary Function (Google Gemini) ---
def generate_ai_summary_gemini(stock_info_dict, latest_ratios_series):
    """
    Generates a financial health summary using Google's Gemini API.
    """
    try:
        api_key = GOOGLE_API_KEY 
        if not api_key: 
            if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
                api_key = st.secrets.get("GOOGLE_API_KEY")
        
        if not api_key:
            return "Error: Google API key not found. Please set it as an environment variable (GOOGLE_API_KEY) or in st.secrets."

        genai.configure(api_key=api_key)
        model_name = 'gemini-1.5-flash-latest'
        model = genai.GenerativeModel(model_name)

        company_name = get_safe_value(stock_info_dict, ['companyName', 'name'], 'the company')
        sector = get_safe_value(stock_info_dict, ['sector'], 'N/A')

        ratios_text_list = []
        if latest_ratios_series is not None and not latest_ratios_series.empty:
            for ratio_name, value in latest_ratios_series.dropna().items():
                if "Margin" in ratio_name or "ROE" in ratio_name or "ROA" in ratio_name:
                    ratios_text_list.append(f"- {ratio_name}: {format_value(value, 'percent')}")
                elif "EPS" in ratio_name:
                    ratios_text_list.append(f"- {ratio_name}: {format_value(value, 'currency_precise')}")
                else:
                    ratios_text_list.append(f"- {ratio_name}: {format_value(value, 'ratio')}")
        ratios_as_text = "\n".join(ratios_text_list) if ratios_text_list else "No ratio data available."

        prompt = f"""
        You are a financial analyst generating a summary for a client. It is MANDATORY that you refer to the company being analyzed by its full name, **{company_name}**, in your analysis. Do not use generic terms like "the company". For example, if the company is "Apple Inc.", you MUST refer to it as "Apple Inc." throughout your response.

        Your task is to analyze the provided key metrics and financial ratios for **{company_name}**, which operates in the **{sector}** sector.
        Your primary analysis MUST be based on the financial data provided below.

        However, you should also leverage your general knowledge about **{company_name}** and its industry. If you see a data point that seems unusual or contradicts your understanding of the company's typical business model (e.g., a surprisingly low ratio for a tech company, or an unusually high one for a utility), you MUST address it. Explain why the number might be misleading or how it should be interpreted in the context of the company's actual operations.

        Provide a brief, well-structured summary (around 150-200 words) of **{company_name}**'s financial health.

        Key Rules to Follow:
        1. **MANDATORY: Use the name "{company_name}" throughout your response.**
        2. Highlight 2-3 key strengths and 2-3 potential areas for attention based on the data and your contextual knowledge.
        3. Acknowledge if the provided data is missing key metrics required for a full assessment.
        4. Do NOT provide any investment advice, buy/sell/hold recommendations, or future price predictions.
        5. Ensure your response is well-formatted for direct display using standard paragraph breaks.

        Key Metrics from Stock Information:
        - Market Cap: {format_value(get_safe_value(stock_info_dict, ['mktCap']), 'currency')}
        - Trailing P/E: {format_value(get_safe_value(stock_info_dict, ['trailingPE', 'peRatioTTM', 'peRatio']), 'ratio')}
        - Forward P/E: {format_value(get_safe_value(stock_info_dict, ['forwardPERatio']), 'ratio')}
        - Beta: {format_value(get_safe_value(stock_info_dict, ['beta']), 'ratio')}

        Latest Financial Ratios:
        {ratios_as_text}

        Please provide your analytical summary for **{company_name}**:
        """

        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=800  # Increased from 400 to prevent summary from being cut off
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        raw_summary = "".join(part.text for part in response.parts) if response.parts else getattr(response, 'text', "")

        if not raw_summary:
            return "Error: AI summary received was empty."
        
        paragraph_placeholder = "[[PARAGRAPH_BREAK_PLACEHOLDER]]"
        cleaned_summary = raw_summary.replace('\n\n', paragraph_placeholder)
        cleaned_summary = cleaned_summary.replace('\n', '')
        cleaned_summary = cleaned_summary.replace(paragraph_placeholder, '\n\n')
        cleaned_summary = re.sub(r' +', ' ', cleaned_summary).strip()
        return cleaned_summary

    except Exception as e:
        st.error(f"An error occurred while generating AI summary with Gemini: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return "Error: Could not generate AI summary with Gemini. Check console for details."

# --- Step 1: Fetch Financial Data (Multi-Year) & Stock Info using FMP ---
@st.cache_data(ttl=3600)
def fetch_financial_data_multi_year_fmp(ticker_symbol, frequency='annual', num_periods=5):
    stock_info_dict_from_profile = {}
    bs_df, is_df, cf_df, hist_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    shares_float_debug_data = None 
    alpha_vantage_overview_debug_data = None 

    try:
        # 1. Fetch Company Profile
        profile_url = f"{FMP_BASE_URL}/v3/profile/{ticker_symbol}?apikey={FMP_API_KEY}"
        response_profile = requests.get(profile_url, timeout=10)
        response_profile.raise_for_status()
        profile_data = response_profile.json()

        if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
            st.error(f"No profile data found for {ticker_symbol} from FMP.")
            return None, bs_df, is_df, cf_df, hist_df, shares_float_debug_data, alpha_vantage_overview_debug_data
        stock_info_dict_from_profile = profile_data[0].copy() 

        temp_stock_info = stock_info_dict_from_profile.copy()

        # 2. Attempt to fetch TTM ratios 
        try:
            ratios_ttm_url = f"{FMP_BASE_URL}/v3/ratios-ttm/{ticker_symbol}?apikey={FMP_API_KEY}"
            response_ratios_ttm = requests.get(ratios_ttm_url, timeout=10)
            response_ratios_ttm.raise_for_status()
            ratios_ttm_data = response_ratios_ttm.json()
            if ratios_ttm_data and isinstance(ratios_ttm_data, list) and len(ratios_ttm_data) > 0:
                temp_stock_info.update(ratios_ttm_data[0]) 
        except Exception as e_ratios:
            st.sidebar.warning(f"Could not fetch TTM ratios for {ticker_symbol}: {str(e_ratios)[:100]}")

        # 3. Attempt to fetch TTM key metrics
        try:
            key_metrics_ttm_url = f"{FMP_BASE_URL}/v3/key-metrics-ttm/{ticker_symbol}?apikey={FMP_API_KEY}"
            response_key_metrics_ttm = requests.get(key_metrics_ttm_url, timeout=10)
            response_key_metrics_ttm.raise_for_status()
            key_metrics_ttm_data = response_key_metrics_ttm.json()
            if key_metrics_ttm_data and isinstance(key_metrics_ttm_data, list) and len(key_metrics_ttm_data) > 0:
                temp_stock_info.update(key_metrics_ttm_data[0])
        except Exception as e_key_metrics:
            st.sidebar.warning(f"Could not fetch TTM key metrics for {ticker_symbol}: {str(e_key_metrics)[:100]}")
        
        # 4. Attempt to fetch Shares Float data for outstanding shares
        try:
            shares_float_url = f"{FMP_BASE_URL}/v3/shares-float/{ticker_symbol}?apikey={FMP_API_KEY}"
            response_shares_float = requests.get(shares_float_url, timeout=10)
            response_shares_float.raise_for_status()
            shares_float_json = response_shares_float.json()
            shares_float_debug_data = shares_float_json 
            if shares_float_json and isinstance(shares_float_json, list) and len(shares_float_json) > 0:
                temp_stock_info.update(shares_float_json[0]) 
        except Exception as e_shares_float:
            st.sidebar.warning(f"Could not fetch shares float data for {ticker_symbol} from FMP: {str(e_shares_float)[:100]}")
            shares_float_debug_data = {"error_fmp_shares_float": str(e_shares_float)}


        # Map FMP fields to a common structure for stock_info_dict
        final_stock_info = {} 
        final_stock_info['shortName'] = get_safe_value(temp_stock_info, ['companyName', 'name'])
        final_stock_info['longBusinessSummary'] = get_safe_value(temp_stock_info, ['description'])
        final_stock_info['currentPrice'] = get_safe_value(temp_stock_info, ['price'])
        final_stock_info['regularMarketPrice'] = get_safe_value(temp_stock_info, ['price'])
        final_stock_info['marketCap'] = get_safe_value(temp_stock_info, ['mktCap', 'marketCapTTM'])
        final_stock_info['regularMarketDayLow'] = get_safe_value(temp_stock_info, ['dayLow'])
        final_stock_info['regularMarketDayHigh'] = get_safe_value(temp_stock_info, ['dayHigh'])
        final_stock_info['fiftyTwoWeekLow'] = get_safe_value(temp_stock_info, ['yearLow'])
        final_stock_info['fiftyTwoWeekHigh'] = get_safe_value(temp_stock_info, ['yearHigh'])
        
        final_stock_info['trailingPE'] = get_safe_value(temp_stock_info, ['peRatioTTM', 'peRatio'])
        final_stock_info['trailingEps'] = get_safe_value(temp_stock_info, ['epsTTM', 'netIncomePerShareTTM', 'eps']) 
        
        final_stock_info['dividendYield'] = get_safe_value(temp_stock_info, ['dividendYieldTTM', 'dividendYield'])
        final_stock_info['payoutRatio'] = get_safe_value(temp_stock_info, ['payoutRatioTTM', 'payoutRatio'])
        final_stock_info['beta'] = get_safe_value(temp_stock_info, ['beta'])
        
        final_stock_info['sharesOutstanding'] = get_safe_value(temp_stock_info, ['outstandingShares', 'sharesOutstanding', 'weightedAverageShsOutDilTTM'])

        # Fallback to Alpha Vantage for Shares Outstanding if still missing
        if (final_stock_info.get('sharesOutstanding') is None or pd.isna(final_stock_info.get('sharesOutstanding'))) and ALPHA_VANTAGE_API_KEY:
            try:
                av_overview_url = f"{ALPHA_VANTAGE_BASE_URL}?function=OVERVIEW&symbol={ticker_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                response_av = requests.get(av_overview_url, timeout=10)
                response_av.raise_for_status()
                av_data = response_av.json()
                alpha_vantage_overview_debug_data = av_data 

                if av_data and isinstance(av_data, dict):
                    av_shares_outstanding_str = av_data.get('SharesOutstanding')
                    if av_shares_outstanding_str and av_shares_outstanding_str != "None" and av_shares_outstanding_str != "0":
                        try:
                            shares_outstanding_av = float(av_shares_outstanding_str)
                            final_stock_info['sharesOutstanding'] = shares_outstanding_av
                        except (ValueError, TypeError):
                            st.sidebar.warning(f"Alpha Vantage 'SharesOutstanding' for {ticker_symbol} was not a valid number: {av_shares_outstanding_str}")
                    elif 'Note' in av_data: 
                        st.sidebar.warning(f"Alpha Vantage API Note for {ticker_symbol} (OVERVIEW): {av_data.get('Note')}")
                    else:
                        st.sidebar.warning(f"SharesOutstanding not found in Alpha Vantage OVERVIEW response for {ticker_symbol}.")
            except requests.exceptions.HTTPError as http_err_av:
                st.sidebar.error(f"Alpha Vantage HTTP Error (OVERVIEW) for {ticker_symbol}: {http_err_av}")
                alpha_vantage_overview_debug_data = {"error_av_http": str(http_err_av)}
            except requests.exceptions.RequestException as req_err_av:
                st.sidebar.error(f"Alpha Vantage Request Error (OVERVIEW) for {ticker_symbol}: {req_err_av}")
                alpha_vantage_overview_debug_data = {"error_av_request": str(req_err_av)}
            except Exception as e_av:
                st.sidebar.error(f"Error fetching/processing Alpha Vantage OVERVIEW data for {ticker_symbol}: {str(e_av)[:100]}")
                alpha_vantage_overview_debug_data = {"error_av_general": str(e_av)}
        elif final_stock_info.get('sharesOutstanding') is None or pd.isna(final_stock_info.get('sharesOutstanding')):
             st.sidebar.warning("ALPHA_VANTAGE_API_KEY not set. Cannot use Alpha Vantage as fallback for Shares Outstanding.")


        final_stock_info['forwardPE'] = get_safe_value(temp_stock_info, ['forwardPERatio', 'priceEarningsToGrowthRatio']) 
        final_stock_info['forwardEps'] = get_safe_value(temp_stock_info, ['epsNextYear']) 

        for key in ['sector', 'industry', 'country', 'website', 'symbol']:
            if key in temp_stock_info: 
                final_stock_info[key] = temp_stock_info[key]
        
        stock_info_dict_to_return = final_stock_info 

        # Fetch Financial Statements
        fmp_period = 'annual' if frequency == 'annual' else 'quarter'
        statement_params = {'period': fmp_period, 'limit': num_periods, 'apikey': FMP_API_KEY}

        for stmt_type, df_var_name in [('income-statement', 'is_df'), 
                                       ('balance-sheet-statement', 'bs_df'), 
                                       ('cash-flow-statement', 'cf_df')]:
            url = f"{FMP_BASE_URL}/v3/{stmt_type}/{ticker_symbol}"
            response = requests.get(url, params=statement_params, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            if json_data:
                df_raw = pd.DataFrame(json_data)
                if not df_raw.empty and 'date' in df_raw.columns:
                    for col in df_raw.columns:
                        if col not in ['date', 'symbol', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate', 'period', 'link', 'finalLink']:
                            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
                    temp_df = df_raw.set_index('date').T
                    temp_df.columns = pd.to_datetime(temp_df.columns).strftime('%Y-%m-%d') 
                    
                    if stmt_type == 'income-statement': is_df = temp_df
                    elif stmt_type == 'balance-sheet-statement': bs_df = temp_df
                    elif stmt_type == 'cash-flow-statement': cf_df = temp_df
        
        if bs_df.empty and is_df.empty and cf_df.empty:
             st.warning(f"No {frequency} financial statement data found for **{ticker_symbol}** from FMP. Analysis may be limited.")

        # Fetch Historical Prices
        to_date = datetime.today().strftime('%Y-%m-%d')
        from_date = (datetime.today() - timedelta(days=4*365)).strftime('%Y-%m-%d')
        hist_params = {'from': from_date, 'to': to_date, 'apikey': FMP_API_KEY}
        hist_url = f"{FMP_BASE_URL}/v3/historical-price-full/{ticker_symbol}"
        response_hist = requests.get(hist_url, params=hist_params, timeout=15)
        response_hist.raise_for_status()
        hist_json = response_hist.json()

        if hist_json and 'historical' in hist_json and hist_json['historical']:
            hist_df_raw = pd.DataFrame(hist_json['historical'])
            if not hist_df_raw.empty and 'date' in hist_df_raw.columns:
                hist_df = hist_df_raw.copy() 
                hist_df['date'] = pd.to_datetime(hist_df['date'])
                hist_df = hist_df.set_index('date').sort_index()
                if 'close' in hist_df.columns and 'Close' not in hist_df.columns:
                    hist_df['Close'] = hist_df['close']
            else:
                st.warning(f"Historical price data for {ticker_symbol} from FMP was empty or malformed.")
        else:
            st.warning(f"No historical price data found for {ticker_symbol} in FMP response.")
        
        return stock_info_dict_to_return, bs_df, is_df, cf_df, hist_df, shares_float_debug_data, alpha_vantage_overview_debug_data

    except requests.exceptions.HTTPError as http_err:
        error_message = f"FMP HTTP Error fetching data for {ticker_symbol}: {http_err}"
        if http_err.response:
            error_message += f" - Status: {http_err.response.status_code} - Response: {http_err.response.text[:200]}"
            if http_err.response.status_code == 401: 
                st.error(f"FMP API Error for {ticker_symbol}: Unauthorized (401). Please check your FMP_API_KEY.")
            elif http_err.response.status_code == 402: 
                 st.error(f"FMP API Error for {ticker_symbol}: Payment Required (402). You might have exceeded free tier limits or are trying to access a premium FMP endpoint.")
            elif http_err.response.status_code == 404: 
                 st.error(f"FMP API Error for {ticker_symbol}: Ticker or endpoint not found (404).")
            elif http_err.response.status_code == 429: 
                st.error(f"FMP API Error for {ticker_symbol}: Too Many Requests (429). You've hit FMP's rate limit. Please wait and try again later.")
            else:
                st.error(error_message)
        else:
            st.error(error_message)
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error_fmp": error_message}, {"error_fmp": "FMP Main Call Failed"}
    except requests.exceptions.RequestException as req_err:
        st.error(f"FMP Request Error fetching data for {ticker_symbol}: {req_err}")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error_fmp": str(req_err)}, {"error_fmp": "FMP Main Call Failed"}
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data for {ticker_symbol} with FMP: {e}")
        st.error(traceback.format_exc())
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"error_fmp": str(e)}, {"error_fmp": "FMP Main Call Failed"}

# --- Step 2: Calculate Financial Ratios (Multi-Year) ---
def calculate_ratios_multi_year(balance_sheet, income_statement, cashflow_statement, stock_info_latest):
    ratios_over_time = pd.DataFrame()
    
    if balance_sheet.empty or income_statement.empty or balance_sheet.columns.empty or income_statement.columns.empty:
        return ratios_over_time

    common_periods = balance_sheet.columns.intersection(income_statement.columns)
    if not cashflow_statement.empty and not cashflow_statement.columns.empty:
        common_periods = common_periods.intersection(cashflow_statement.columns)
    
    if common_periods.empty:
        st.warning("No common periods found between financial statements for ratio calculation.")
        return ratios_over_time
        
    common_periods_dt = pd.to_datetime(common_periods).sort_values(ascending=True)

    for period_col_dt in common_periods_dt: 
        period_col_str = period_col_dt.strftime('%Y-%m-%d') 
        ratios = {}
        
        bs_period = balance_sheet[period_col_str] if period_col_str in balance_sheet.columns else pd.Series(dtype='float64')
        is_period = income_statement[period_col_str] if period_col_str in income_statement.columns else pd.Series(dtype='float64')
        cf_period = cashflow_statement[period_col_str] if not cashflow_statement.empty and period_col_str in cashflow_statement.columns else pd.Series(dtype='float64')

        current_assets = get_safe_value(bs_period, ['totalCurrentAssets'])
        current_liabilities = get_safe_value(bs_period, ['totalCurrentLiabilities'])
        cash_equivalents = get_safe_value(bs_period, ['cashAndCashEquivalents', 'cashAndShortTermInvestments'])
        inventory = get_safe_value(bs_period, ['inventory'])
        accounts_receivable = get_safe_value(bs_period, ['netReceivables'])
        total_liabilities_val = get_safe_value(bs_period, ['totalLiabilities'])
        shareholder_equity = get_safe_value(bs_period, ['totalStockholdersEquity', 'totalEquity'])
        total_assets = get_safe_value(bs_period, ['totalAssets'])
        long_term_debt = get_safe_value(bs_period, ['longTermDebt'])
        short_term_debt = get_safe_value(bs_period, ['shortTermDebt'])
        total_debt = get_safe_value(bs_period, ['totalDebt'])
        if pd.isna(total_debt) or total_debt == 0: 
            calculated_total_debt = (float(long_term_debt) if pd.notna(long_term_debt) else 0) + \
                                    (float(short_term_debt) if pd.notna(short_term_debt) else 0)
            if calculated_total_debt > 0 : total_debt = calculated_total_debt

        ebit = get_safe_value(is_period, ['operatingIncome']) 
        interest_expense_val = get_safe_value(is_period, ['interestExpense'])
        interest_expense = abs(float(interest_expense_val)) if pd.notna(interest_expense_val) else None
        net_income = get_safe_value(is_period, ['netIncome'])
        revenue = get_safe_value(is_period, ['revenue'])
        gross_profit = get_safe_value(is_period, ['grossProfit'])
        cogs = get_safe_value(is_period, ['costOfRevenue'])
        
        ebitda = get_safe_value(is_period, ['ebitda'])
        if pd.isna(ebitda): 
            depreciation_amortization = get_safe_value(is_period, ['depreciationAndAmortization'])
            operating_income_for_ebitda_calc = get_safe_value(is_period, ['operatingIncome'])
            if pd.notna(operating_income_for_ebitda_calc) and pd.notna(depreciation_amortization):
                try:
                    ebitda = float(operating_income_for_ebitda_calc) + float(depreciation_amortization)
                except (ValueError, TypeError): pass

        basic_eps = get_safe_value(is_period, ['eps']) 
        diluted_eps = get_safe_value(is_period, ['epsdiluted'])
        ratios['EPS (Basic)'] = basic_eps
        ratios['EPS (Diluted)'] = diluted_eps if pd.notna(diluted_eps) else basic_eps

        op_cash_flow = get_safe_value(cf_period, ['netCashProvidedByOperatingActivities', 'operatingCashFlow'])
        cap_ex_val = get_safe_value(cf_period, ['capitalExpenditure'])
        cap_ex = abs(float(cap_ex_val)) if pd.notna(cap_ex_val) else None

        def safe_div(num, den):
            if pd.notna(num) and num is not None and pd.notna(den) and den is not None:
                try:
                    num_f = float(num)
                    den_f = float(den)
                    if den_f != 0:
                        return num_f / den_f
                except (ValueError, TypeError): return None
            return None

        ratios['Current Ratio'] = safe_div(current_assets, current_liabilities)
        ratios['Quick Ratio (Cash based)'] = safe_div(cash_equivalents, current_liabilities)
        if pd.notna(current_assets) and pd.notna(inventory):
            try:
                ratios['Quick Ratio (Excl. Inventory)'] = safe_div((float(current_assets) - float(inventory)), current_liabilities)
            except (ValueError, TypeError): pass
        
        ratios['Debt-to-Equity'] = safe_div(total_debt, shareholder_equity)
        ratios['Debt-to-Assets'] = safe_div(total_liabilities_val, total_assets)
        ratios['Financial Leverage (Assets/Equity)'] = safe_div(total_assets, shareholder_equity)
        ratios['Interest Coverage Ratio (EBIT/Int)'] = safe_div(ebit, interest_expense)
        ratios['Net Profit Margin'] = safe_div(net_income, revenue)
        ratios['Gross Profit Margin'] = safe_div(gross_profit, revenue)
        ratios['EBITDA Margin'] = safe_div(ebitda, revenue)
        ratios['Operating Margin'] = safe_div(ebit, revenue) 
        ratios['ROA (Return on Assets)'] = safe_div(net_income, total_assets)
        ratios['ROE (Return on Equity)'] = safe_div(net_income, shareholder_equity)
        ratios['Asset Turnover'] = safe_div(revenue, total_assets)
        
        inv_turnover = safe_div(cogs, inventory)
        if inv_turnover is not None:
            ratios['Inventory Turnover'] = inv_turnover
            if inv_turnover != 0: ratios['Days Inventory Outstanding (DIO)'] = 365 / inv_turnover
        
        rec_turnover = safe_div(revenue, accounts_receivable)
        if rec_turnover is not None:
            ratios['Receivables Turnover'] = rec_turnover
            if rec_turnover != 0: ratios['Days Sales Outstanding (DSO)'] = 365 / rec_turnover
        
        if pd.notna(op_cash_flow) and pd.notna(cap_ex):
            try:
                ratios['Free Cash Flow (FCF)'] = float(op_cash_flow) - float(cap_ex)
            except (ValueError, TypeError): pass

        ratios_over_time[period_col_str] = pd.Series(ratios) 

    if not ratios_over_time.empty:
        for col_name in ratios_over_time.columns:
            ratios_over_time[col_name] = pd.to_numeric(ratios_over_time[col_name], errors='coerce')
            if pd.api.types.is_numeric_dtype(ratios_over_time[col_name]): 
                 ratios_over_time[col_name] = ratios_over_time[col_name].astype(np.float64) 
            
    return ratios_over_time.sort_index(axis=1, ascending=False)

# --- DCF Calculation Function ---
def calculate_dcf(stock_info, latest_financials, ratios_df,
                  risk_free_rate, market_risk_premium, cost_of_debt_pre_tax,
                  fcf_growth_rate_short_term, perpetual_growth_rate, projection_years,
                  base_fcf_override=None):
    """
    Performs a simplified Discounted Cash Flow (DCF) analysis.
    Allows override of base FCF for projection.
    """
    dcf_results = {}
    
    # 1. Gather Inputs
    beta = get_safe_value(stock_info, ['beta'])
    market_cap = get_safe_value(stock_info, ['marketCap', 'mktCap']) 
    shares_outstanding = get_safe_value(stock_info, ['sharesOutstanding', 'weightedAverageShsOutDilTTM', 'outstandingShares']) 
    
    if base_fcf_override is not None:
        latest_fcf = base_fcf_override
    else:
        latest_fcf = get_safe_value(ratios_df[ratios_df.columns[0]], ['Free Cash Flow (FCF)']) if not ratios_df.empty and ratios_df.columns.any() else None
    
    total_debt = get_safe_value(latest_financials, ['totalDebt'])
    cash_and_equivalents = get_safe_value(latest_financials, ['cashAndCashEquivalents', 'cashAndShortTermInvestments'])
    income_tax_expense = get_safe_value(latest_financials, ['incomeTaxExpense'])
    income_before_tax = get_safe_value(latest_financials, ['incomeBeforeTax'])

    # Check for essential data
    essential_data_missing = []
    if beta is None: essential_data_missing.append("Beta")
    if market_cap is None: essential_data_missing.append("Market Cap")
    if shares_outstanding is None: essential_data_missing.append("Shares Outstanding")
    if latest_fcf is None: essential_data_missing.append("Latest FCF")
    if total_debt is None: essential_data_missing.append("Total Debt")
    if cash_and_equivalents is None: essential_data_missing.append("Cash & Equivalents")
    
    if essential_data_missing:
        dcf_results['error'] = f"DCF Error: Missing essential data: {', '.join(essential_data_missing)}."
        return dcf_results

    try:
        beta = float(beta)
        market_cap = float(market_cap)
        shares_outstanding = float(shares_outstanding)
        latest_fcf = float(latest_fcf)
        total_debt = float(total_debt)
        cash_and_equivalents = float(cash_and_equivalents)
    except (ValueError, TypeError) as e:
        dcf_results['error'] = f"DCF Error: Could not convert essential financial data to numbers: {e}"
        return dcf_results

    # 2. Calculate Effective Tax Rate
    effective_tax_rate = 0.21 
    if pd.notna(income_tax_expense) and pd.notna(income_before_tax) and income_before_tax is not None:
        try:
            val_income_before_tax = float(income_before_tax)
            if val_income_before_tax != 0:
                effective_tax_rate = float(income_tax_expense) / val_income_before_tax
                effective_tax_rate = max(0, min(effective_tax_rate, 1)) 
            elif float(income_tax_expense) == 0: 
                 effective_tax_rate = 0.0
        except (ValueError, TypeError):
            st.sidebar.warning("Could not calculate effective tax rate from financials, using default 21%.")
    else:
        st.sidebar.warning("Income tax expense or income before tax not found for tax rate calculation, using default 21%.")
    dcf_results['effective_tax_rate'] = effective_tax_rate
    
    # 3. Calculate Cost of Equity (Ke)
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    dcf_results['cost_of_equity'] = cost_of_equity
    
    # 4. Calculate WACC
    ev_for_wacc = market_cap + total_debt - cash_and_equivalents
    if ev_for_wacc == 0: 
        dcf_results['error'] = "DCF Error: Calculated EV for WACC (Market Cap + Total Debt - Cash) is zero. Cannot proceed."
        return dcf_results

    equity_ratio = market_cap / ev_for_wacc
    debt_ratio = total_debt / ev_for_wacc 
    
    wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt_pre_tax * (1 - effective_tax_rate))
    dcf_results['wacc'] = wacc

    if wacc <= perpetual_growth_rate:
        dcf_results['error'] = f"DCF Error: WACC ({wacc:.2%}) must be greater than the perpetual growth rate ({perpetual_growth_rate:.2%}) for terminal value calculation."
        return dcf_results
    if wacc <= 0: 
        dcf_results['error'] = f"DCF Error: Calculated WACC ({wacc:.2%}) is not positive. Check inputs."
        return dcf_results


    # 5. Project FCFs and Discount them
    projected_fcfs = []
    pv_fcfs = []
    current_fcf = latest_fcf
    
    growth_rates_list = [fcf_growth_rate_short_term] * int(projection_years) 

    for i in range(1, int(projection_years) + 1):
        growth = growth_rates_list[i-1]
        current_fcf *= (1 + growth)
        projected_fcfs.append(current_fcf)
        pv_fcf = current_fcf / ((1 + wacc) ** i)
        pv_fcfs.append(pv_fcf)
    
    dcf_results['projected_fcfs'] = projected_fcfs
    sum_pv_fcfs = sum(pv_fcfs)
    dcf_results['pv_fcfs_sum'] = sum_pv_fcfs
    
    # 6. Calculate Terminal Value and its Present Value
    terminal_fcf = projected_fcfs[-1] 
    terminal_value = (terminal_fcf * (1 + perpetual_growth_rate)) / (wacc - perpetual_growth_rate)
    pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
    
    dcf_results['terminal_value_at_projection_end'] = terminal_value
    dcf_results['pv_terminal_value'] = pv_terminal_value
    
    # 7. Calculate Enterprise Value and Equity Value
    enterprise_value = sum_pv_fcfs + pv_terminal_value
    net_debt = total_debt - cash_and_equivalents 
    equity_value = enterprise_value - net_debt
    
    dcf_results['enterprise_value'] = enterprise_value
    dcf_results['net_debt'] = net_debt
    dcf_results['equity_value'] = equity_value
    
    # 8. Calculate Intrinsic Value per Share
    if shares_outstanding == 0:
        dcf_results['error'] = "DCF Error: Shares outstanding is zero."
        dcf_results['intrinsic_value_per_share'] = "N/A"
        return dcf_results
        
    intrinsic_value_per_share = equity_value / shares_outstanding
    dcf_results['intrinsic_value_per_share'] = intrinsic_value_per_share
    
    return dcf_results

# --- Step 3: Loan Scenario Modeling --- (and other existing functions)
def simulate_loan_impact(latest_financial_data, loan_amount, interest_rate_decimal):
    results = {'New Annual Interest': None, 'Pro-Forma Interest Expense': None,
               'Pro-Forma Interest Coverage Ratio': None, 'Acceptable Coverage (>1.25x)?': None}
    if not latest_financial_data: 
        st.warning("Loan simulation: essential financial data (EBIT, Interest Expense) missing from latest income statement period.")
        return results

    ebit = get_safe_value(latest_financial_data, ['operatingIncome']) 
    current_interest_expense_val = get_safe_value(latest_financial_data, ['interestExpense'], default=0.0)
    current_interest_expense = abs(float(current_interest_expense_val)) if pd.notna(current_interest_expense_val) else 0.0

    if pd.notna(ebit) and ebit is not None: 
        try:
            ebit_float = float(ebit)
            annual_interest_increase = loan_amount * interest_rate_decimal
            new_total_interest_expense = current_interest_expense + annual_interest_increase
            results['New Annual Interest'] = annual_interest_increase
            results['Pro-Forma Interest Expense'] = new_total_interest_expense
            if new_total_interest_expense != 0:
                new_icr = ebit_float / new_total_interest_expense
                results['Pro-Forma Interest Coverage Ratio'] = new_icr
                results['Acceptable Coverage (>1.25x)?'] = new_icr > 1.25
            else:
                if ebit_float > 0:
                    results['Pro-Forma Interest Coverage Ratio'] = float('inf')
                    results['Acceptable Coverage (>1.25x)?'] = True
                elif ebit_float < 0:
                    results['Pro-Forma Interest Coverage Ratio'] = float('-inf')
                    results['Acceptable Coverage (>1.25x)?'] = False
                else:
                    results['Pro-Forma Interest Coverage Ratio'] = 0.0
                    results['Acceptable Coverage (>1.25x)?'] = False
        except (ValueError, TypeError):
             st.warning("Error converting EBIT to float for loan simulation.")
    else:
        st.warning("EBIT (operatingIncome) not found in latest financial data for loan simulation.")
    return results

def assess_risk_from_ratios(latest_ratios_series):
    score = 0
    thresholds = {'Current Ratio': 1.5, 'Debt-to-Equity': 2.0, 'Interest Coverage Ratio (EBIT/Int)': 3.0,
                  'Net Profit Margin': 0.05, 'EBITDA Margin': 0.10,
                  'Quick Ratio (Excl. Inventory)': 1.0,
                  'ROA (Return on Assets)': 0.05}

    achieved_criteria = []
    max_score = len(thresholds)

    if latest_ratios_series is None or latest_ratios_series.empty:
        return 'N/A', 'grey', "Not enough data for risk assessment."

    def check_ratio(name, threshold, is_greater_better=True):
        nonlocal score
        val = get_safe_value(latest_ratios_series, name) 
        if pd.notnull(val) and val is not None:
            try:
                val_float = float(val)
                formatted_val_display = f"{val_float:.2f}"
                if any(term in name for term in ["Margin", "ROE", "ROA", "Yield"]):
                    formatted_val_display = f"{val_float:.2%}"

                if (is_greater_better and val_float >= threshold) or \
                   (not is_greater_better and val_float <= threshold):
                    score += 1
                    achieved_criteria.append(f"{name} ({formatted_val_display} {'(Good)' if is_greater_better else '(Manageable)'})")
                else:
                    achieved_criteria.append(f"{name} ({formatted_val_display} {'(Below Thresh.)' if is_greater_better else '(Above Thresh.)'})")
            except (ValueError, TypeError):
                achieved_criteria.append(f"{name} (N/A - format error)")
    for name, thresh in thresholds.items():
        check_ratio(name, thresh, is_greater_better=(name != 'Debt-to-Equity'))

    if score >= max_score * 0.7: risk_level, color = 'Low Risk', 'green'
    elif score >= max_score * 0.4: risk_level, color = 'Medium Risk', 'orange'
    else: risk_level, color = 'High Risk', 'red'
    summary_text = f"**Risk Score:** {score}/{max_score}. "
    summary_text += "--- ".join(achieved_criteria) if achieved_criteria else "No specific ratio strengths/weaknesses identified."
    return risk_level, color, summary_text

def filter_financial_statement_for_display(df_statement):
    if df_statement is None or df_statement.empty: return pd.DataFrame()
    df_display = df_statement.copy()
    for col_name in df_display.columns:
        df_display[col_name] = pd.to_numeric(df_display[col_name], errors='coerce')
        if pd.api.types.is_numeric_dtype(df_display[col_name]): # Check if it's numeric after coercion
            df_display[col_name] = df_display[col_name].astype(np.float64) # Ensure float for NaNs
    rows_to_keep = df_display.apply(lambda row: not (row.isnull().all() or (abs(row.fillna(0)) < 1e-6).all()), axis=1)
    return df_display[rows_to_keep]

def display_formatted_statement(df, statement_name):
    st.subheader(f"{statement_name} (Key Items)")
    display_df = filter_financial_statement_for_display(df)
    if not display_df.empty:
        df_to_display = display_df.head(30).copy()
        # First, rename the index to be user-friendly
        df_to_display.index = df_to_display.index.map(snake_case_to_title)
        
        # Then, create the Styler object with the new, readable index
        # The format_statement_values function will now work on the readable names
        styled_df = format_statement_values(df_to_display)
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info(f"{statement_name} data not available or all items are zero/N/A from FMP.")

# --- Streamlit App UI ---
st.title("Financial Analysis Toolkit")
st.caption("Comprehensive financial analysis using Financial Modeling Prep API. All monetary values are in the currency reported by FMP (usually USD unless specified).")

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")
ticker_df = load_ticker_file() 
ticker_options_display = [''] 
ticker_symbol_map = {}

if not ticker_df.empty:
    for index, row in ticker_df.iterrows():
        display_label = row['Display Label']
        ticker_options_display.append(display_label)
        ticker_symbol_map[display_label] = row['Symbol']
else:
    st.sidebar.warning("Could not load ticker list from FMP. Auto-completion may not work or be limited.")

selected_ticker_display = st.sidebar.selectbox(
    "Select or Type Stock Ticker (e.g., AAPL - Apple Inc.):",
    options=ticker_options_display, index=0,
    help="Start typing a ticker or company name. List dynamically fetched from FMP."
)
ticker = ""
if selected_ticker_display:
    if selected_ticker_display in ticker_symbol_map:
        ticker = ticker_symbol_map[selected_ticker_display]
    else:
        ticker = selected_ticker_display.split(" - ")[0].strip().upper()
elif st.session_state.get('last_successful_ticker'):
    ticker = st.session_state['last_successful_ticker']

frequency_options = {'Annual': 'annual', 'Quarterly': 'quarterly'}
selected_frequency_label = st.sidebar.radio("Select Data Frequency:", list(frequency_options.keys()), index=0, key="data_frequency_radio")
frequency_value = frequency_options[selected_frequency_label]

# --- Sidebar Expander for Scenarios ---
with st.sidebar.expander("Valuation & Scenario Modeling", expanded=False):
    st.subheader("DCF Assumptions")
    dcf_projection_years = st.number_input("Projection Years for DCF:", min_value=3, max_value=10, value=5, step=1, key="dcf_proj_years")
    dcf_fcf_growth_rate = st.slider("FCF Growth Rate (Short-Term, % per year):", -10.0, 25.0, 5.0, 0.1, format="%.1f%%", key="dcf_fcf_growth") / 100 # Allow negative growth
    dcf_perpetual_growth_rate = st.slider("Perpetual Growth Rate (%):", 0.0, 5.0, 2.5, 0.1, format="%.1f%%", key="dcf_perp_growth") / 100

    # Advanced DCF Inputs (collapsible)
    with st.container(): # Use a container to group these
        if "show_advanced_dcf" not in st.session_state:
            st.session_state.show_advanced_dcf = False

        def toggle_advanced_dcf():
            st.session_state.show_advanced_dcf = not st.session_state.show_advanced_dcf

        st.checkbox("Show Advanced DCF Inputs", value=st.session_state.show_advanced_dcf, key="adv_dcf_toggle", on_change=toggle_advanced_dcf)

        if st.session_state.show_advanced_dcf:
            dcf_risk_free_rate = st.slider("Risk-Free Rate (e.g., 10-yr Treasury, %):", 0.0, 10.0, 4.2, 0.1, format="%.1f%%", key="dcf_rf") / 100
            dcf_market_risk_premium = st.slider("Market Risk Premium (%):", 0.0, 15.0, 5.5, 0.1, format="%.1f%%", key="dcf_mrp") / 100
            dcf_cost_of_debt = st.slider("Pre-Tax Cost of Debt (%):", 0.0, 15.0, 5.0, 0.1, format="%.1f%%", key="dcf_cod") / 100
        else: # Default values if advanced inputs are hidden
            dcf_risk_free_rate = 0.042 
            dcf_market_risk_premium = 0.055
            dcf_cost_of_debt = 0.050

    # FCF Base Method Selection
    st.subheader("FCF Base for DCF")
    fcf_base_method = st.radio(
        "How should the base Free Cash Flow (FCF) be determined?",
        options=["Latest Period", "Average (3 Years)", "Manual Entry"],
        index=1, # Default to 3-year average
        help="Using an average of historical FCF can smooth out single-year anomalies. 'Latest Period' uses the most recent FCF, which might be volatile."
    )

    manual_fcf_input = None
    if fcf_base_method == "Manual Entry":
        manual_fcf_input = st.number_input(
            "Manual FCF ($, millions):",
            min_value=0.0, value=1000.0, step=100.0, key="manual_fcf_sidebar"
        ) * 1_000_000


    st.subheader("Loan Scenario Details")
    loan_amount = st.number_input("Loan Amount ($):", min_value=0, value=100000, step=10000, key="loan_amount_sidebar_input_fmp")
    interest_rate_perc = st.slider("Annual Interest Rate (%):", 0.0, 25.0, 7.0, 0.1, key="interest_rate_sidebar_slider_fmp")
    interest_rate_dec = interest_rate_perc / 100.0

analyze_button = st.sidebar.button("Analyze Company")

# --- Main App Logic ---
if analyze_button and ticker:
    st.session_state['last_successful_ticker'] = ticker
    # This function call will now trigger the data fetching
    # and the result will be stored in session state
    with st.spinner(f"Fetching and analyzing **{ticker}** ({selected_frequency_label})..."):
        stock_info, bs_data, is_data, cf_data, stock_hist, shares_float_debug_data, alpha_vantage_overview_debug_data = fetch_financial_data_multi_year_fmp(ticker, frequency_value)
        
        # Store all fetched data in session state so it persists across reruns
        st.session_state['data_cache'] = {
            'ticker': ticker,
            'frequency_value': frequency_value,
            'stock_info': stock_info,
            'bs_data': bs_data,
            'is_data': is_data,
            'cf_data': cf_data,
            'stock_hist': stock_hist,
            'shares_float_debug_data': shares_float_debug_data,
            'alpha_vantage_overview_debug_data': alpha_vantage_overview_debug_data
        }

# Use cached data if it exists for the current ticker/frequency
# This part of the logic runs on every interaction (e.g., slider change)
if ticker and 'data_cache' in st.session_state and st.session_state['data_cache']['ticker'] == ticker and st.session_state['data_cache']['frequency_value'] == frequency_value:
    
    # Retrieve all data from the session state cache
    cached_data = st.session_state['data_cache']
    stock_info = cached_data.get('stock_info')
    bs_data = cached_data.get('bs_data')
    is_data = cached_data.get('is_data')
    cf_data = cached_data.get('cf_data')
    stock_hist = cached_data.get('stock_hist')
    shares_float_debug_data = cached_data.get('shares_float_debug_data')
    alpha_vantage_overview_debug_data = cached_data.get('alpha_vantage_overview_debug_data')

    if stock_info and get_safe_value(stock_info, ['shortName', 'companyName', 'name']): 
        st.header(f"{get_safe_value(stock_info, ['shortName', 'companyName', 'name'], ticker)} ({get_safe_value(stock_info, ['symbol'], ticker)}) - {selected_frequency_label} Data Overview")

        ratios_df = calculate_ratios_multi_year(bs_data, is_data, cf_data, stock_info)
        
        # --- Prepare data and checks for DCF ---
        base_fcf_for_dcf = None
        if fcf_base_method == "Latest Period":
            base_fcf_for_dcf = get_safe_value(ratios_df.iloc[:, 0], ['Free Cash Flow (FCF)']) if not ratios_df.empty and ratios_df.columns.any() else None
        elif fcf_base_method == "Average (3 Years)":
            base_fcf_for_dcf = get_normalized_fcf(ratios_df, periods=3)
        elif fcf_base_method == "Manual Entry":
            base_fcf_for_dcf = manual_fcf_input
        
        dcf_prerequisites = {
            "Base FCF": base_fcf_for_dcf,
            "Beta": get_safe_value(stock_info, ['beta']),
            "Market Cap": get_safe_value(stock_info, ['marketCap', 'mktCap']),
            "Shares Outstanding": get_safe_value(stock_info, ['sharesOutstanding', 'weightedAverageShsOutDilTTM', 'outstandingShares']), 
            "Total Debt (Latest)": None,
            "Cash & Equivalents (Latest)": None,
            "Income Tax Expense (Latest)": None,
            "Income Before Tax (Latest)": None
        }
        latest_financials_for_dcf = {}

        if not bs_data.empty and not bs_data.columns.empty:
            latest_bs_col = bs_data.columns[0] 
            dcf_prerequisites["Total Debt (Latest)"] = get_safe_value(bs_data[latest_bs_col], ['totalDebt'])
            latest_financials_for_dcf['totalDebt'] = dcf_prerequisites["Total Debt (Latest)"]
            dcf_prerequisites["Cash & Equivalents (Latest)"] = get_safe_value(bs_data[latest_bs_col], ['cashAndCashEquivalents', 'cashAndShortTermInvestments'])
            latest_financials_for_dcf['cashAndCashEquivalents'] = dcf_prerequisites["Cash & Equivalents (Latest)"]
        
        if not is_data.empty and not is_data.columns.empty:
            latest_is_col = is_data.columns[0] 
            dcf_prerequisites["Income Tax Expense (Latest)"] = get_safe_value(is_data[latest_is_col], ['incomeTaxExpense'])
            latest_financials_for_dcf['incomeTaxExpense'] = dcf_prerequisites["Income Tax Expense (Latest)"]
            dcf_prerequisites["Income Before Tax (Latest)"] = get_safe_value(is_data[latest_is_col], ['incomeBeforeTax'])
            latest_financials_for_dcf['incomeBeforeTax'] = dcf_prerequisites["Income Before Tax (Latest)"]

        missing_prerequisites = [key for key, value in dcf_prerequisites.items() if value is None or pd.isna(value)]
        can_run_dcf_updated = not missing_prerequisites
        
        # --- Tab List Names Logic ---
        tabs_list_names = ["Key Metrics & Stock Info", "Historical Financials", "Financial Ratios (YoY)"]
        
        dcf_tab_name_to_add = None
        if can_run_dcf_updated:
            dcf_tab_name_to_add = "DCF Valuation"
        elif missing_prerequisites: 
            dcf_tab_name_to_add = "DCF Valuation (Data Missing)"
        
        if dcf_tab_name_to_add:
            tabs_list_names.append(dcf_tab_name_to_add)

        if ratios_df is not None and not ratios_df.empty:
            tabs_list_names.append("AI Financial Summary")
        tabs_list_names.append("Loan Impact & Risk Assessment")
        
        created_tabs = st.tabs(tabs_list_names)
        tab_idx = 0

        # --- Key Metrics & Stock Info Tab ---
        if "Key Metrics & Stock Info" in tabs_list_names and tab_idx < len(created_tabs):
            with created_tabs[tab_idx]: 
                tab_idx += 1
                # ... [This UI section is the same as the previous version] ...
                st.subheader("Company Overview")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sector", get_safe_value(stock_info, ['sector'], 'N/A'))
                col2.metric("Industry", get_safe_value(stock_info, ['industry'], 'N/A'))
                col3.metric("Country", get_safe_value(stock_info, ['country'], 'N/A'))
                website = get_safe_value(stock_info, ['website'])
                if website: st.markdown(f"**Website:** [{website}]({website})")
                summary_text_val = get_safe_value(stock_info, ['longBusinessSummary','description'], 'N/A') 
                st.markdown(f"**Business Summary:** {summary_text_val[:500]}{'...' if len(summary_text_val) > 500 else ''}")
                st.subheader("Current Stock Performance")
                perf_cols = st.columns(4)
                current_price_val = get_safe_value(stock_info, ['currentPrice', 'price'])
                if current_price_val is not None and pd.notna(current_price_val):
                    perf_cols[0].metric("Current Price", format_value(current_price_val, 'currency_precise'))
                market_cap_val = get_safe_value(stock_info, ['marketCap', 'mktCap'])
                if market_cap_val is not None and pd.notna(market_cap_val):
                    perf_cols[1].metric("Market Cap", format_value(market_cap_val, 'currency'))
                day_low_val = get_safe_value(stock_info, ['regularMarketDayLow', 'dayLow'])
                day_high_val = get_safe_value(stock_info, ['regularMarketDayHigh', 'dayHigh'])
                if (day_low_val is not None and pd.notna(day_low_val)) or \
                   (day_high_val is not None and pd.notna(day_high_val)):
                    perf_cols[2].metric("Day Low / High", f"{format_value(day_low_val, 'currency_precise')} / {format_value(day_high_val, 'currency_precise')}")
                w52_low_val = get_safe_value(stock_info, ['fiftyTwoWeekLow', 'yearLow'])
                w52_high_val = get_safe_value(stock_info, ['fiftyTwoWeekHigh', 'yearHigh'])
                if (w52_low_val is not None and pd.notna(w52_low_val)) or \
                   (w52_high_val is not None and pd.notna(w52_high_val)):
                    perf_cols[3].metric("52 Week Low / High", f"{format_value(w52_low_val, 'currency_precise')} / {format_value(w52_high_val, 'currency_precise')}")
                st.subheader("Valuation & Dividends")
                val_cols = st.columns(4)
                trailing_pe_val = get_safe_value(stock_info, ['trailingPE', 'peRatioTTM', 'peRatio'])
                price = get_safe_value(stock_info, ['currentPrice', 'price'])
                trailing_eps = get_safe_value(stock_info, ['trailingEps', 'epsTTM', 'netIncomePerShareTTM', 'eps'])
                if (trailing_pe_val is None or pd.isna(trailing_pe_val)) and price and trailing_eps:
                    try:
                        price_float = float(price)
                        eps_float = float(trailing_eps)
                        if eps_float > 0: trailing_pe_val = price_float / eps_float
                    except (ValueError, TypeError): pass 
                if trailing_pe_val is not None and pd.notna(trailing_pe_val):
                    val_cols[0].metric("Trailing P/E", format_value(trailing_pe_val, 'ratio'))
                forward_pe_val = get_safe_value(stock_info, ['forwardPE', 'forwardPERatio'])
                if forward_pe_val is not None and pd.notna(forward_pe_val):
                    val_cols[0].metric("Forward P/E", format_value(forward_pe_val, 'ratio'))
                if trailing_eps is not None and pd.notna(trailing_eps):
                    val_cols[1].metric("Trailing EPS", format_value(trailing_eps, 'currency_precise'))
                forward_eps_val = get_safe_value(stock_info, ['forwardEps', 'epsNextYear'])
                if forward_eps_val is not None and pd.notna(forward_eps_val):
                    val_cols[1].metric("Forward EPS", format_value(forward_eps_val, 'currency_precise'))
                raw_div_yield = get_safe_value(stock_info, ['dividendYield', 'dividendYieldTTM']) 
                if pd.isna(raw_div_yield) or raw_div_yield == 0:
                    last_annual_dividend = get_safe_value(stock_info, ['lastAnnualDividend', 'dividendLastAnnual']) 
                    if pd.notna(last_annual_dividend) and pd.notna(price):
                        try:
                            price_float = float(price)
                            if price_float != 0:
                                raw_div_yield = float(last_annual_dividend) / price_float
                        except (ValueError, TypeError): raw_div_yield = None 
                if raw_div_yield is not None and pd.notna(raw_div_yield):
                    val_cols[2].metric("Dividend Yield", format_value(raw_div_yield, 'percent'))
                payout_ratio_val = get_safe_value(stock_info, ['payoutRatio', 'payoutRatioTTM'])
                if payout_ratio_val is not None and pd.notna(payout_ratio_val):
                     val_cols[2].metric("Payout Ratio", format_value(payout_ratio_val, 'percent'))
                beta_val = get_safe_value(stock_info, ['beta'])
                if beta_val is not None and pd.notna(beta_val):
                    val_cols[3].metric("Beta", format_value(beta_val, 'ratio'))
                shares_outstanding_val = get_safe_value(stock_info, ['sharesOutstanding', 'weightedAverageShsOutDilTTM', 'outstandingShares'])
                if shares_outstanding_val is not None and pd.notna(shares_outstanding_val):
                    val_cols[3].metric("Shares Outstanding", format_value(shares_outstanding_val, 'number'))
                if stock_hist is not None and not stock_hist.empty and 'Close' in stock_hist.columns:
                    st.subheader("Historical Stock Price (Last 4 Years)")
                    fig_price = px.line(stock_hist, x=stock_hist.index, y='Close', title=f'{ticker} Close Price Over Time')
                    fig_price.update_layout(xaxis_title="Date", yaxis_title="Close Price ($)")
                    st.plotly_chart(fig_price, use_container_width=True)
                elif stock_hist is not None and not stock_hist.empty:
                    st.warning("Historical price data fetched, but 'Close' column not found as expected. Check FMP data structure.")


        # --- Historical Financials Tab ---
        if "Historical Financials" in tabs_list_names and tab_idx < len(created_tabs):
            with created_tabs[tab_idx]:
                tab_idx += 1
                display_formatted_statement(bs_data, "Balance Sheet")
                display_formatted_statement(is_data, "Income Statement")
                display_formatted_statement(cf_data, "Cash Flow Statement")


        # --- Financial Ratios Tab ---
        if "Financial Ratios (YoY)" in tabs_list_names and tab_idx < len(created_tabs):
            with created_tabs[tab_idx]:
                tab_idx += 1
                st.subheader("Key Financial Ratios Over Time")
                if ratios_df is not None and not ratios_df.empty:
                    # The format_statement_values function can be used here as well for consistent formatting
                    st.dataframe(format_statement_values(ratios_df))
                    st.subheader("Ratio Trends")
                    default_ratios = ['Current Ratio', 'Debt-to-Equity', 'Net Profit Margin', 'ROE (Return on Equity)', 'EPS (Diluted)']
                    available_ratios = [r for r in default_ratios if r in ratios_df.index]
                    selected_ratios = st.multiselect("Select ratios to plot:", options=list(ratios_df.index), default=available_ratios, key="ratio_plot_multiselect_fmp")
                    if selected_ratios:
                        plot_df = ratios_df.loc[selected_ratios].T.reset_index().rename(columns={'index': 'Period'})
                        try:
                            plot_df['Period'] = pd.to_datetime(plot_df['Period'])
                            plot_df = plot_df.sort_values(by='Period')
                        except Exception: st.warning("Could not parse period dates for optimal chart sorting.")
                        for ratio_name_plot in selected_ratios:
                            if ratio_name_plot in plot_df.columns and not plot_df[ratio_name_plot].dropna().empty: 
                                fig = px.line(plot_df, x='Period', y=ratio_name_plot, title=f'{ratio_name_plot} Trend', markers=True)
                                if any(p_term in ratio_name_plot for p_term in ["Margin", "ROE", "ROA", "Yield"]) and "EPS" not in ratio_name_plot :
                                    fig.update_layout(yaxis_tickformat='.2%')
                                fig.update_layout(xaxis_title=f"{selected_frequency_label} Period Ending")
                                st.plotly_chart(fig, use_container_width=True)
                            elif ratio_name_plot in plot_df.columns:
                                st.info(f"No data available to plot for {ratio_name_plot}.")
                    else: st.info("No ratios selected for plotting.")
                else: st.info("Ratio data not available for plotting from FMP.")

        # --- DCF Valuation Tab ---
        if dcf_tab_name_to_add and tab_idx < len(created_tabs): 
            with created_tabs[tab_idx]:
                tab_idx += 1
                st.subheader(dcf_tab_name_to_add) 
                st.markdown("""
                **Disclaimer:** This DCF model is a simplified educational tool and uses several assumptions. 
                It should NOT be considered financial advice. Always conduct thorough research and consult professionals.
                Intrinsic value is highly sensitive to input assumptions.
                """)

                if can_run_dcf_updated:
                    with st.expander("Show DCF Calculation Trace", expanded=False):
                        calculate_dcf(
                            stock_info, latest_financials_for_dcf, ratios_df,
                            dcf_risk_free_rate, dcf_market_risk_premium, dcf_cost_of_debt,
                            dcf_fcf_growth_rate, dcf_perpetual_growth_rate, dcf_projection_years,
                            base_fcf_override=base_fcf_for_dcf 
                        )
                    
                    dcf_output = calculate_dcf(
                        stock_info, latest_financials_for_dcf, ratios_df,
                        dcf_risk_free_rate, dcf_market_risk_premium, dcf_cost_of_debt,
                        dcf_fcf_growth_rate, dcf_perpetual_growth_rate, dcf_projection_years,
                        base_fcf_override=base_fcf_for_dcf
                    )

                    if 'error' in dcf_output:
                        st.error(dcf_output['error'])
                    else:
                        st.markdown("#### DCF Results:")
                        col_dcf1, col_dcf2 = st.columns(2)
                        col_dcf1.metric("Calculated WACC", format_value(dcf_output.get('wacc'), 'percent'))
                        col_dcf1.metric("Calculated Cost of Equity", format_value(dcf_output.get('cost_of_equity'), 'percent'))
                        col_dcf1.metric("Calculated Effective Tax Rate", format_value(dcf_output.get('effective_tax_rate'), 'percent'))
                        
                        col_dcf2.metric("Enterprise Value (EV)", format_value(dcf_output.get('enterprise_value'), 'currency'))
                        col_dcf2.metric("Equity Value", format_value(dcf_output.get('equity_value'), 'currency'))
                        col_dcf2.metric("Intrinsic Value per Share", format_value(dcf_output.get('intrinsic_value_per_share'), 'currency_precise'))
                        
                        current_stock_price = get_safe_value(stock_info, ['currentPrice', 'price'])
                        if current_stock_price is not None and dcf_output.get('intrinsic_value_per_share') is not None:
                            try:
                                current_price_float = float(current_stock_price)
                                intrinsic_value_float = float(dcf_output['intrinsic_value_per_share']) 
                                if pd.notna(current_price_float) and pd.notna(intrinsic_value_float) and current_price_float != 0: 
                                    diff = (intrinsic_value_float - current_price_float) / current_price_float
                                    col_dcf2.metric("Upside/Downside vs Current Price", format_value(diff, 'percent'))
                                elif current_price_float == 0:
                                     col_dcf2.metric("Upside/Downside vs Current Price", "N/A (Current Price is 0)")

                            except (ValueError, TypeError, KeyError): pass 

                        st.markdown("#### Projected Free Cash Flows:")
                        if 'projected_fcfs' in dcf_output and isinstance(dcf_output['projected_fcfs'], list):
                            fcf_proj_df = pd.DataFrame({
                                'Year': [f"Year {i+1}" for i in range(len(dcf_output['projected_fcfs']))],
                                'Projected FCF': dcf_output['projected_fcfs']
                            })
                            st.dataframe(fcf_proj_df.style.format({'Projected FCF': lambda x: format_value(x, 'currency')}))
                        
                        st.markdown(f"**Terminal Value (at end of Year {int(dcf_projection_years)}):** {format_value(dcf_output.get('terminal_value_at_projection_end'), 'currency')}")
                        st.markdown(f"**Present Value of Terminal Value:** {format_value(dcf_output.get('pv_terminal_value'), 'currency')}")
                        st.markdown(f"**Sum of PV of Projected FCFs:** {format_value(dcf_output.get('pv_fcfs_sum'), 'currency')}")
                else: 
                    st.warning("Cannot run DCF valuation due to missing data. Required items not found for this ticker:")
                    for item in missing_prerequisites:
                        st.markdown(f"- {item}")
                    st.markdown("Please check if FMP provides this data for the selected ticker on your API plan, or if the data field names need adjustment in the code. See Debug Info in sidebar for details.")


        # --- AI Financial Summary Tab ---
        if "AI Financial Summary" in tabs_list_names and tab_idx < len(created_tabs):
            with created_tabs[tab_idx]:
                tab_idx += 1
                st.subheader("AI Financial Health Summary (via Google Gemini)")
                st.markdown("""**Disclaimer:** This AI-generated summary is for informational purposes only and is not financial advice. It's based on the provided data and may not capture all nuances. Always do your own research.""")
                if ratios_df is not None and not ratios_df.empty and not ratios_df.columns.empty: 
                    latest_ratios = ratios_df[ratios_df.columns[0]]
                    with st.spinner("Generating AI summary..."):
                        ai_summary = generate_ai_summary_gemini(stock_info, latest_ratios)
                        st.markdown(ai_summary)
                else: st.info("Financial ratios are needed to generate an AI summary.")
        
        # --- Loan Impact & Risk Assessment Tab ---
        if "Loan Impact & Risk Assessment" in tabs_list_names and tab_idx < len(created_tabs): 
            with created_tabs[tab_idx]:
                st.subheader("Loan Impact Simulation (Based on Latest Data)")
                loan_sim_inputs = {}
                if isinstance(is_data, pd.DataFrame) and not is_data.empty and not is_data.columns.empty:
                    latest_is_col_name = is_data.columns[0] 
                    is_latest_period_data = is_data[latest_is_col_name]
                    loan_sim_inputs['operatingIncome'] = get_safe_value(is_latest_period_data, ['operatingIncome']) 
                    interest_val = get_safe_value(is_latest_period_data, ['interestExpense'])
                    loan_sim_inputs['interestExpense'] = abs(float(interest_val)) if pd.notna(interest_val) else 0.0
                else:
                    st.warning("Income statement data for the latest period is unavailable for loan simulation (FMP).")

                if loan_amount > 0 and pd.notna(get_safe_value(loan_sim_inputs, ['operatingIncome'])):
                    loan_results = simulate_loan_impact(loan_sim_inputs, loan_amount, interest_rate_dec)
                    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                    col_l1.metric("New Annual Interest", format_value(loan_results.get('New Annual Interest'), 'currency'))
                    col_l2.metric("Pro-Forma Total Int. Exp.", format_value(loan_results.get('Pro-Forma Interest Expense'), 'currency'))
                    pro_forma_icr = loan_results.get('Pro-Forma Interest Coverage Ratio')
                    col_l3.metric("Pro-Forma Int. Coverage", f"{pro_forma_icr:.2f}x" if pd.notna(pro_forma_icr) and pro_forma_icr not in [float('inf'), float('-inf')] else "N/A")
                    accept_cov = loan_results.get('Acceptable Coverage (>1.25x)?')
                    col_l4.metric("Coverage Acceptable (>1.25x)?", "Yes" if accept_cov is True else ("No" if accept_cov is False else "N/A"))
                elif loan_amount > 0:
                    st.warning("Could not perform loan simulation. Ensure EBIT (operatingIncome) data is available from FMP for the latest period.")
                else:
                    st.info("Enter a loan amount > 0 in the sidebar to simulate impact.")

                st.subheader("Overall Risk Assessment (Based on Latest Ratios)")
                if ratios_df is not None and not ratios_df.empty and not ratios_df.columns.empty:
                    latest_ratios_series = ratios_df[ratios_df.columns[0]]
                    risk_level, risk_color, risk_summary = assess_risk_from_ratios(latest_ratios_series)
                    st.markdown(f"**<font color='{risk_color}'>{risk_level}</font>**", unsafe_allow_html=True)
                    st.markdown(f"**Detailed Breakdown:** {risk_summary}")
                else: st.info("Risk assessment requires calculated ratios. Ensure data was fetched and ratios calculated from FMP.")


    elif ticker and not (stock_info and get_safe_value(stock_info, ['shortName', 'companyName', 'name'])):
        if not stock_info: 
             st.error(f"Could not retrieve any valid data for the ticker: **{ticker}** using FMP. Please check the ticker or FMP API status/limits.")
        st.session_state['last_successful_ticker'] = None
    elif not ticker and analyze_button:
        st.error("Please enter a stock ticker in the sidebar or select one from the dropdown.")
    elif not ticker:
        st.info("Welcome! Please enter a stock ticker in the sidebar (or select from the dropdown) and click 'Analyze Company'.")
