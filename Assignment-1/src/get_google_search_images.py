import os
import math
import argparse
import urllib.request
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from googleapiclient.discovery import build
from utils import create_dir, styled_print
load_dotenv()

API_KEY = os.getenv('GCP_API_KEY')
SEARCH_ENGINE_ID = os.getenv('GCP_SEARCH_ENGINE_ID')
OUT_DATA_DIR = "data/google-search-images"
NUM_RESULTS = 100
calls_to_make = math.ceil(NUM_RESULTS / 10)


def build_service(api_key):
    service = build("customsearch", "v1", developerKey=api_key)
    styled_print("Created service object ...")
    return service


def search_images(service, search_engine_id, query, start=1, num_results=10, imageType='photo', urls=[]):
    styled_print(f"Searching Images using {query}", header=True)
    if start != 1:
        start = ((start-1) * 10) + 1
    results = service.cse().list(
        q=query,
        cx=search_engine_id,
        start=start,
        searchType='image',
        num=num_results,
        imgType=imageType,
        fileType='png',
        safe='off'
    ).execute()

    if not 'items' in results:
        print(f'No result !!\nres is: {results}')
        pass
    else:
        urls.extend([i['link'] for i in results['items']])
    styled_print(f"Found {len(urls)} URLs.")
    return results, urls


def download_images(urls, out_dir, verbose=0):
    image_dict = defaultdict(list)
    styled_print(f"Downloading Images ...", header=True)
    out_dir = create_dir(os.path.abspath(os.getcwd()), out_dir, header=False)
    for i, url in enumerate(urls):
        filepath = os.path.join(out_dir, f"{str(i)}.png")
        styled_print(f"Downloading {url} to {filepath}")
        try:
            urllib.request.urlretrieve(
                url,
                filepath
            )
            image_dict["url"].append(url)
            image_dict["file_path"].append(filepath)
        except:
            continue
    return image_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Google Search Downloader')
    parser.add_argument('--query', type=str, nargs='+',
                        help='Type your test query.')
    args = parser.parse_args()

    styled_print("Initiating Google Custom Search Downloader", header=True)
    service = build_service(API_KEY)

    for q in args.query:
        styled_print(f"Working on `{q}` Query ...", header=True)
        query = q
        urls = []
        for i in range(1, calls_to_make+1):
            search_results, urls = search_images(
                service,
                search_engine_id=SEARCH_ENGINE_ID,
                start=i,
                query=query,
                num_results=10,
                imageType='photo',
                urls=urls
            )

        if urls is not None:
            output_dir = create_dir(OUT_DATA_DIR, query.replace(" ", "-"))
            image_data_dict = download_images(urls, output_dir)
            image_data_dict["query"] = [
                query for i in range(len(image_data_dict["url"]))]

        if os.path.isfile(os.path.join(OUT_DATA_DIR, 'data.csv')):
            data_df = pd.read_csv(os.path.join(OUT_DATA_DIR, 'data.csv'))
            df = pd.DataFrame(image_data_dict)
            output = pd.concat([data_df, df], ignore_index=True)
            output.to_csv(os.path.join(OUT_DATA_DIR, 'data.csv'),
                          index=False, header=True)
        else:
            pd.DataFrame(image_data_dict).to_csv(os.path.join(OUT_DATA_DIR, 'data.csv'),
                                                 index=False, header=True)
