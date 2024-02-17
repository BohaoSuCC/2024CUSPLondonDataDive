import requests
import json
import os



def load_data_from_api(url, data_format='xml', filename=None):
    """
    Fetches data from a remote API and saves it to a local file with a custom file name.
    
    Parameters:
    - url (str): The URL of the remote API.
    - data_format (str): The format of the data ('xml' or 'json'). Defaults to 'xml'.
    - filename (str): The custom filename for saving the data. If None, defaults to 'local_data'.
    
    Returns:
    - None
    """
    # check if the data_format is valid and the filename has the correct extension
    file_extension = data_format.lower()
    if file_extension not in ['xml', 'json']:
        raise ValueError("data_format must be 'xml' or 'json'")
    
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
        # save the data to a local file
        if data_format == 'json':
            data = response.json()
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        else:
            with open(filename, 'wb') as file:
                file.write(response.content)
        print(f"Data saved to {filename}.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
