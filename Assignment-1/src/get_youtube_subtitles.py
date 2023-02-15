import os
import json
import argparse
from collections import defaultdict
from youtubesearchpython import VideosSearch
from utils import create_dir, styled_print


OUT_DATA_DIR = "../data/processed-data/search-images"


def youtube_search(query, limit):
    list_links = defaultdict(list)
    videos_search = VideosSearch(query, limit)
    search_results = videos_search.result()
    for res in search_results["result"]:
        list_links[res["title"]] = res["link"]
    return list_links


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YouTube Parser')
    parser.add_argument('--query', type=str, nargs='+',
                        help="Queries for YouTube Search.")
    parser.add_argument('--num-videos', type=int,
                        help="Number of YouTube videos to Extract.")
    parser.add_argument('--config-file', type=str,
                        default=None, help="Path to JSON file")
    args = parser.parse_args()

    styled_print("Initiating YouTube Extractor", header=True)

    with open(args.config_file, 'r') as f:
        hod_data = json.load(f)

    for q in args.query:
        styled_print(
            f"Extracting {args.num_videos} YouTube Videos for {q}", header=True)
        youtube_links = youtube_search(q, args.num_videos)
        id = 1
        for k, v in youtube_links.items():
            styled_print(f"Found YouTube Video {k} at {v}")
            hod_data["youtube"].append(
                {
                    "id": id,
                    "title": k,
                    "url": v
                }
            )
            id += 1
