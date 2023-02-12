import requests
from bs4 import BeautifulSoup
import os

QUERY = "cats"
MIN_WIDTH = 0
MIN_HEIGHT = 0


def search_images(query, min_width, min_height):
    """Sends a search request to Bing Images and returns a list of image URLs."""
    # Format the search query for use in the URL
    query = query.replace(" ", "+")

    # Send the search request
    response = requests.get(
        f"https://www.bing.com/images/search?q={query}&FORM=HDRSC2")

    # Check if the request was successful
    if response.status_code == 200:
        # Use BeautifulSoup to parse the HTML response
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all the image thumbnail elements
        image_elements = soup.find_all("img")

        # Extract the image URLs and dimensions from the elements
        image_urls = []
        for element in image_elements:
            width = int(element.get("width", 0))
            height = int(element.get("height", 0))

            if width >= min_width and height >= min_height:
                image_urls.append(element["src"])

        return image_urls
    else:
        # Return an empty list if the request was unsuccessful
        print("Search request failed.")
        return []


def download_images(image_urls, folder):
    """Downloads images to the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, url in enumerate(image_urls):
        response = requests.get(url)

        if response.status_code == 200:
            with open(f"{folder}/image_{i}.jpg", "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image from URL: {url}")


# Run the script
if __name__ == "__main__":
    image_urls = search_images(QUERY, MIN_WIDTH, MIN_HEIGHT)
    download_images(image_urls, "images")
