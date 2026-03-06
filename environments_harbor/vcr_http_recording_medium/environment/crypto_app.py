#!/usr/bin/env python3

import requests
from typing import Dict, List, Optional

BASE_API_URL = 'https://api.coinbase.com/v2/prices'


def get_crypto_price(crypto_code: str) -> Dict[str, str]:
    """
    Fetch cryptocurrency price from Coinbase API.
    
    Args:
        crypto_code: Cryptocurrency code (e.g., 'BTC-USD', 'ETH-USD')
    
    Returns:
        Dictionary containing price data with 'amount' and 'currency' fields
    
    Raises:
        requests.exceptions.RequestException: If network error occurs
        ValueError: If invalid response received
    """
    try:
        url = f"{BASE_API_URL}/{crypto_code}/spot"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' not in data:
            raise ValueError(f"Invalid response format for {crypto_code}")
        
        price_data = data['data']
        
        return {
            'amount': price_data.get('amount'),
            'currency': price_data.get('currency'),
            'base': price_data.get('base')
        }
    
    except requests.exceptions.Timeout:
        raise requests.exceptions.RequestException(f"Timeout fetching price for {crypto_code}")
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.RequestException(f"Connection error fetching price for {crypto_code}")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.RequestException(f"HTTP error fetching price for {crypto_code}: {e}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error parsing response for {crypto_code}: {e}")


def get_multiple_prices(crypto_list: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Fetch prices for multiple cryptocurrencies.
    
    Args:
        crypto_list: List of cryptocurrency codes
    
    Returns:
        Dictionary mapping crypto codes to their price data
    """
    results = {}
    
    for crypto_code in crypto_list:
        try:
            price_data = get_crypto_price(crypto_code)
            results[crypto_code] = price_data
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error fetching {crypto_code}: {e}")
            results[crypto_code] = {'error': str(e)}
    
    return results


if __name__ == '__main__':
    print("Fetching cryptocurrency prices...\n")
    
    # Fetch single price
    try:
        btc_price = get_crypto_price('BTC-USD')
        print(f"Bitcoin (BTC-USD): ${btc_price['amount']} {btc_price['currency']}")
    except Exception as e:
        print(f"Error fetching BTC price: {e}")
    
    print()
    
    # Fetch multiple prices
    crypto_codes = ['BTC-USD', 'ETH-USD']
    prices = get_multiple_prices(crypto_codes)
    
    print("Multiple cryptocurrency prices:")
    for code, data in prices.items():
        if 'error' in data:
            print(f"{code}: Error - {data['error']}")
        else:
            print(f"{code}: ${data['amount']} {data['currency']}")