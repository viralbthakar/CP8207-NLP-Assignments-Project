import os
import json
import pysrt
import argparse
import trafilatura
import pandas as pd
from collections import defaultdict
from utils import create_dir, styled_print

OUT_DATA_DIR = "../data/processed-data/url-texts"


def extract_text_from_srt(file_path):
    subtitle_dict = defaultdict(list)
    subtitles = pysrt.open(file_path)
    for sub in subtitles:
        subtitle_dict["start_hours"].append(sub.start.hours)
        subtitle_dict["start_minutes"].append(sub.start.minutes)
        subtitle_dict["start_seconds"].append(sub.start.seconds)
        subtitle_dict["end_hours"].append(sub.end.hours)
        subtitle_dict["end_minutes"].append(sub.end.minutes)
        subtitle_dict["end_seconds"].append(sub.end.seconds)
        subtitle_dict["text"].append(sub.text)
    return subtitle_dict


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

    styled_print("Initiating Data Extractor", header=True)

    if args.hod_urls_file is not None:
        styled_print(
            "Extracting Data for House of Dragons Season", header=True)
        out_dir = create_dir(OUT_DATA_DIR, "house-of-dragons", header=False)

        with open(args.hod_urls_file, 'r') as f:
            hod_data = json.load(f)

        for key in hod_data["house-of-dragons"].keys():
            styled_print(
                f"Working on {key} of House of Dragons", header=True)
            element_dir = create_dir(out_dir, key, header=False)
            elements = {epi["title"]: epi["url"] for epi in
                        hod_data["house-of-dragons"][key]}
            styled_print(
                f"Found {len(elements.keys())} {key} ...", header=True)
            for title, url in elements.items():
                styled_print(f"Extracting {title} {key} ...")
                if key == "subtitles":
                    subtitle_dict = extract_text_from_srt(url)
                    styled_print(
                        f"Writing Subtitles into {title.replace(' ', '-')}.csv ...")
                    pd.DataFrame(subtitle_dict).to_csv(
                        os.path.join(
                            element_dir, f"{title.replace(' ', '-')}.csv"),
                        index=False, header=True)
                else:
                    text = extract_text_from_url(url)
                    styled_print(
                        f"Writing Text into {title.replace(' ', '-')}.txt ...")
                    with open(os.path.join(element_dir, f"{title.replace(' ', '-')}.txt"), 'w') as f:
                        f.write(text)
