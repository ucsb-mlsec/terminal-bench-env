#!/usr/bin/env python3

import sys
import requests

def main():
    url = "http://localhost:8080/health"
    timeout = 0.15  # 150ms timeout
    
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            sys.exit(0)
        else:
            sys.exit(1)
    except requests.exceptions.Timeout:
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        sys.exit(1)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()