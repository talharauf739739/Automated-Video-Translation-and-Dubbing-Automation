from pytube import YouTube
import os

def download_video(video_url, output_path="mp4_videos/"):
    """
    Download a YouTube video in MP4 format.

    Args:
        video_url (str): YouTube video URL.
        output_path (str): Path to save the downloaded video.

    Returns:
        str: Path to the downloaded MP4 file.
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Get YouTube video
        yt = YouTube(video_url)

        # Filter for MP4 video with the highest resolution
        video_stream = yt.streams.filter(file_extension="mp4", progressive=True).get_highest_resolution()

        if not video_stream:
            print("No MP4 video stream available.")
            return None

        print(f"Downloading video: {yt.title}")
        
        # Download video
        video_file = video_stream.download(output_path)
        print(f"Downloaded and saved to: {video_file}")
        
        return video_file
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

if __name__ == "__main__":
    # Use a simple URL without extra parameters
    video_url = 'https://www.youtube.com/watch?v=ckWgWPnbTq8'

    # Download the video
    downloaded_video = download_video(video_url)

    if downloaded_video:
        print(f"Video successfully downloaded: {downloaded_video}")
    else:
        print("Video download failed.")
