#!/usr/bin/env python3

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import json
import time

def main():
    # Set up Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to the local HTML file
        driver.get('file:///home/user/task/index.html')
        
        # Wait for the page to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
        
        # Give the page a moment to fully render
        time.sleep(2)
        
        # Extract product data with stale element handling
        products = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Re-locate table rows on each attempt to avoid stale references
                rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')
                
                temp_products = []
                for row in rows:
                    try:
                        # Extract data immediately to minimize chance of staleness
                        cells = row.find_elements(By.TAG_NAME, 'td')
                        if len(cells) >= 2:
                            product_name = cells[0].text
                            product_price = cells[1].text
                            temp_products.append({
                                'name': product_name,
                                'price': product_price
                            })
                    except StaleElementReferenceException:
                        # If a single row becomes stale, break and retry the whole extraction
                        raise StaleElementReferenceException("Row became stale")
                
                # If we successfully extracted all rows, use this data
                products = temp_products
                break
                
            except StaleElementReferenceException:
                if attempt < max_retries - 1:
                    # Wait a moment and retry
                    time.sleep(0.5)
                    continue
                else:
                    # On final attempt, just use what we have
                    pass
        
        # Write results to JSON file
        result = {
            "count": len(products),
            "status": "success"
        }
        
        with open('/home/user/task/results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Successfully extracted {len(products)} products")
        
    finally:
        # Clean up
        driver.quit()

if __name__ == '__main__':
    main()