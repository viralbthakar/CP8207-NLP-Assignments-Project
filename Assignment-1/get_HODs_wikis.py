import os
import json
import argparse
import trafilatura
from utils import create_dir, styled_print

OUT_DATA_DIR = "data/url-texts"


def extract_text_from_url(url):
    # Download HTML Code
    downloaded_url = trafilatura.fetch_url(url)

    # Try Extracting Text data as
    try:
        extract = trafilatura.extract(
            downloaded_url,
            output_format='json',
            favor_precisions=True,
            favour_recall=True,
            include_comments=False,
            include_tables=False,
            date_extraction_params={
                'extensive_search': True, 'original_date': True}
        )
    except AttributeError:
        extract = trafilatura.extract(
            downloaded_url,
            output_format='json',
            date_extraction_params={
                'extensive_search': True, 'original_date': True}
        )
    if extract:
        json_output = json.loads(extract)
        return json_output['text']
    else:
        return "None"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wikis Parser')
    parser.add_argument('--urls', type=str, nargs='+', default=None,
                        help='Type your test urls.')
    parser.add_argument('--hod-urls-file', type=str,
                        default=None, help="Path to JSON file")
    args = parser.parse_args()

    styled_print("Initiating Wikis Parser", header=True)

    if args.hod_urls_file is not None:
        styled_print(
            "Extracting Wikis for House of Dragons Episodes", header=True)
        out_dir = create_dir(OUT_DATA_DIR, "hod", header=False)

        with open(args.hod_urls_file, 'r') as f:
            hod_data = json.load(f)

        styled_print(f"{json.dumps(hod_data, indent=4)}")

        for key in hod_data["house-of-dragons"].keys():
            element_dir = create_dir(out_dir, key, header=False)
            elements = {epi["title"]: epi["url"] for epi in
                        hod_data["house-of-dragons"][key]}

            styled_print(
                f"Found {len(elements.keys())} {key} ...", header=True)

            for title, url in elements.items():
                styled_print(f"Extracting {title} {key} ...")
                episode_dir = create_dir(
                    element_dir, title.replace(' ', '-'), header=False)
                text = extract_text_from_url(url)
                with open(os.path.join(episode_dir, f"{title.replace(' ', '-')}.txt"), 'w') as f:
                    f.write(text)
