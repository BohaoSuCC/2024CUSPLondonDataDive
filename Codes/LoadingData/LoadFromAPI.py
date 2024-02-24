import requests
import json
import csv
import os

def load_data_from_api(url, data_format='xml', filename=None):
    """
    Fetches data from a remote API and saves it to a local file with a custom file name.
    
    Parameters:
    - url (str): The URL of the remote API.
    - data_format (str): The format of the data ('xml', 'json', or 'csv'). Defaults to 'xml'.
    - filename (str): The custom filename for saving the data. If None, defaults to 'local_data'.
    
    Returns:
    - None
    """
    # check if the data_format is valid and the filename has the correct extension
    file_extension = data_format.lower()
    if file_extension not in ['xml', 'json', 'csv']:
        raise ValueError("data_format must be 'xml', 'json', or 'csv'")
    
    # if there is no filename, use a default filename with the correct extension
    if filename is None:
        filename = f"local_data.{file_extension}"
    else:
        # make sure the filename has the correct extension
        if not filename.endswith(f".{file_extension}"):
            filename += f".{file_extension}"
    
    # set get request to the API
    response = requests.get(url)
    
    # check if the request was successful
    if response.status_code == 200:
        if data_format == 'json':
            # save the data in JSON format
            data = response.json()
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        elif data_format == 'csv':
            # save the data in CSV format
            # This assumes that the data is a list of dictionaries
            data = response.text
            # Convert text to list of dictionaries
            lines = data.splitlines()
            reader = csv.DictReader(lines)
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    writer.writerow(row)
        else:
            # save the data in XML format or any other plain text format
            with open(filename, 'wb') as file:
                file.write(response.content)
        print(f"Data saved to {filename}.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
