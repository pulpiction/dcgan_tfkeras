import os
import requests
from tqdm import tqdm

"""
Codes here are modified from the assignment for CSCI1470 at Brown University
"""

def download(id, dest): 
    url = 'https://docs.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirmation(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, dest)

def get_confirmation(response):
    for key, val in response.cookies.items():
        if key.startswith('download_warning'):
            return val
    return None

def save_response_content(response, dest, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit='B', unit_sale=True, desc=dest):
        
            if chunk:
                f.write(chunk)
    
def download_celeb(path):
    fn, drive_id = 'celebA.tar.gz', '1gWqvG6E7VXGi257Y-K2tGKBsKaIAuBR2'
    save_path = os.path.join(path, fn)

    if os.path.exists(save_path):
        print('Path already exists ... ')
    else:
        download(drive_id, save_path)

def prepare_data_dir(path='./data'):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    prepare_data_dir()
    download_celeb('./data')