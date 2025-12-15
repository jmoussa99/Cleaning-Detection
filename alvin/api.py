"""
Script to fetch events data from PulseListener API
"""
import requests
import json
from datetime import datetime

def fetch_events(device_id, from_date, to_date):
    """
    Fetch events from the PulseListener API
    
    Args:
        device_id (str): Device ID
        from_date (str): Start date in format YYYY-MM-DD
        to_date (str): End date in format YYYY-MM-DD
    
    Returns:
        dict: JSON response from the API
    """
    # API endpoint
    base_url = "https://pulselistener.azurewebsites.net/api/fetchEvents"
    
    # Parameters
    params = {
        'device': device_id,
        'from': from_date,
        'to': to_date
    }
    
    print(f"Fetching events for device: {device_id}")
    print(f"Date range: {from_date} to {to_date}")
    print(f"URL: {base_url}")
    print(f"Parameters: {params}\n")
    
    try:
        # Make GET request
        response = requests.get(base_url, params=params, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        # Check if response contains data
        if response.status_code == 404:
            # Parse JSON to check if it's a "no data" error
            try:
                error_data = response.json()
                if 'error' in error_data and 'No events found' in error_data.get('error', ''):
                    print(f"⚠️  No events found for the specified date range")
                    print(f"   Device: {error_data.get('device', 'N/A')}")
                    print(f"   Date range: {error_data.get('from', 'N/A')} to {error_data.get('to', 'N/A')}")
                    print(f"\n   This device may not have data for these dates.")
                    print(f"   Try a different date range when the device was active.")
                    return None
            except:
                pass
            print(f"Error: Resource not found (404)")
            return None
        
        # Raise for other HTTP errors
        response.raise_for_status()
        
        print(f"✅ Response received successfully!\n")
        
        # Parse JSON or JSONL response
        content_type = response.headers.get('content-type', '')
        
        # Check if response is JSONL (multiple JSON objects, one per line)
        if 'application/x-ndjson' in content_type or 'jsonl' in content_type:
            # Parse JSONL
            data = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    data.append(json.loads(line))
            return data
        else:
            # Regular JSON
            data = response.json()
            return data
    
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after 30 seconds")
        print(f"   The API request took too long to complete.")
        print(f"   Try reducing the date range or check your network connection.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        return None

def save_to_file(data, filename=None, format='json'):
    """
    Save data to a JSON or JSONL file
    
    Args:
        data: Data to save
        filename (str): Output filename (optional)
        format (str): 'json' or 'jsonl' (default: 'json')
    """
    if data is None:
        print("No data to save")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = 'jsonl' if format == 'jsonl' else 'json'
        filename = f"events_data_{timestamp}.{ext}"
    
    if format == 'jsonl':
        # Save as JSONL (one JSON object per line)
        with open(filename, 'w') as f:
            if isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item) + '\n')
            else:
                f.write(json.dumps(data) + '\n')
    else:
        # Save as regular JSON
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Get file size for reporting
    import os
    file_size = os.path.getsize(filename)
    size_mb = file_size / (1024 * 1024)
    
    print(f"✅ Data saved to: {filename}")
    print(f"   File size: {size_mb:.2f} MB ({file_size:,} bytes)")

if __name__ == "__main__":
    # Configuration
    # Note: According to API docs:
    # - device: Device ID or "all" for all devices
    # - from: Start date (format: YYYY-MM-DD) - required
    # - to: End date (format: YYYY-MM-DD) - optional, defaults to today
    # - Limitation: When device=all, date range cannot exceed 31 days
    
    DEVICE_ID = "e00fce68e6bfa7f7c691797b"
    FROM_DATE = "2025-11-10"  # Date range with known data
    TO_DATE = "2025-12-10"    # Verified working date range
    
    # Fetch data
    data = fetch_events(DEVICE_ID, FROM_DATE, TO_DATE)
    
    if data:
        # Check if data is a list or dict
        if isinstance(data, list):
            print(f"✅ Received {len(data)} events")
        elif isinstance(data, dict):
            print(f"✅ Received data: {len(str(data))} characters")
        
        # Show a preview of the first event
        print(f"\nData preview (first event):")
        if isinstance(data, list) and len(data) > 0:
            print(json.dumps(data[0], indent=2)[:500])
        else:
            print(json.dumps(data, indent=2)[:500])
        print("...\n")
        
        # Save to JSONL file
        output_filename = f"events_{FROM_DATE}_to_{TO_DATE}"
        save_to_file(data, f"{output_filename}.jsonl", format='jsonl')
    else:
        print("❌ Failed to fetch data")