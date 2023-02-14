#Imports
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests
import re
from utils import styled_print


#Search actual youtube query
def youtube_search(query, limit):
    styled_print("Searching Youtube Videos...", header=True)
    list_links = []
    videos_search = VideosSearch(query, limit)
    search_results = videos_search.result()
    for x in search_results["result"]:
        list_links.append(x["link"])
        styled_print(f"Found video: {x['link']}")
    return list_links

#Create json to paste into json file
def create_json(id, youtube_video):
     #Get video title
    reqs = requests.get(youtube_video)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    for title in soup.find_all('title'):
        #Remove - YouTube suffix
        video_title = title.get_text().replace("- YouTube", "").replace('"','\\"')
        styled_print("")
    # print("{")
    # print(f'"id" : {id},\n"title" : "{video_title}",\n"url" : "{youtube_video}"')
    # print("},")

#Search actual youtube query. Store links in `youtube_videos``
youtube_videos = youtube_search(str(input("Enter your search query:")), 30)

transcripts = {}
def retrieve_transcript(id, youtube_video):  
    #Get video title
    reqs = requests.get(youtube_video)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    for title in soup.find_all('title'):
        #Remove - YouTube suffix
        video_title = title.get_text().replace("- YouTube", "")
    
    #Find Youtube ID string
    youtube_video = re.findall(r"v=.{11}", youtube_video)
    youtube_video = youtube_video[0][2:]
    #Extract transcript
    try:
        transcript_json = YouTubeTranscriptApi.get_transcript(youtube_video)
        styled_print(f"Got transcript for video: {video_title}")
    except:
        youtube_videos.pop(id-1)
        return
    #Get Text only
    transcript_text = ""
    for x in transcript_json:
        transcript_text += " "+ x["text"].replace("\n", " ")
    transcripts[id] = {"title": video_title, "text" : transcript_text}


styled_print("Scraping video transcripts...", header=True)

for id, video in enumerate(youtube_videos):
    retrieve_transcript(id+1, video)

# for video in youtube_videos:
#     create_json(youtube_videos.index(video), video)

#Create files
import os.path
from os import path
def create_transcript_json_files(dircetory, name, text):
    transcript_file = open(f"{dircetory}/{name}", "w")
    transcript_file.write(text)
for videoID in transcripts:
    video = transcripts[videoID]
    # if not path.exists(video["title"]):
    create_transcript_json_files("../data/raw-data/youtube-transcripts", "yt_vid"+str(videoID), video["text"])

