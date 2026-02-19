#!/usr/bin/env python3

import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def main():
    # Configure Chrome in headless mode
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Load the dashboard HTML file
        dashboard_path = 'file:///home/user/dashboard_scraper/dashboard.html'
        driver.get(dashboard_path)
        
        # Wait for initial page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'widget'))
        )
        
        snapshots = []
        
        # Collect data at 3 time points
        for snapshot_num in range(3):
            print(f"Collecting snapshot {snapshot_num + 1}...")
            
            # Re-fetch elements each time to avoid stale element references
            widgets = driver.find_elements(By.CLASS_NAME, 'widget')
            
            snapshot_data = []
            for widget in widgets:
                # Extract widget data - re-find child elements to avoid staleness
                widget_name = widget.find_element(By.CLASS_NAME, 'widget-name').text
                widget_value = widget.find_element(By.CLASS_NAME, 'widget-value').text
                
                snapshot_data.append({
                    'name': widget_name,
                    'value': widget_value
                })
            
            snapshots.append({
                'time_point': f'T{snapshot_num}',
                'widgets': snapshot_data
            })
            
            print(f"Snapshot {snapshot_num + 1} collected: {len(snapshot_data)} widgets")
            
            # Wait for next update cycle (except after the last snapshot)
            if snapshot_num < 2:
                time.sleep(2.5)  # Wait for JavaScript to update the DOM
        
        # Verify we collected the right amount of data
        if len(snapshots) == 3 and all(len(s['widgets']) == 5 for s in snapshots):
            result = {
                'snapshots_collected': 3,
                'total_widgets': 5,
                'status': 'success'
            }
        else:
            result = {
                'snapshots_collected': len(snapshots),
                'total_widgets': len(snapshots[0]['widgets']) if snapshots else 0,
                'status': 'failed'
            }
        
        # Write output file
        with open('/home/user/dashboard_scraper/output.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Script completed successfully. Output written to output.json")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        result = {
            'snapshots_collected': 0,
            'total_widgets': 0,
            'status': 'failed'
        }
        with open('/home/user/dashboard_scraper/output.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    finally:
        driver.quit()

if __name__ == '__main__':
    main()