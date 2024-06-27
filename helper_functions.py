import cv2
import numpy as np

def resize_video(width, height, max_width = 600):
  # max_width = in pixels. define the maximum width of the processed video.
  # the height will be proportional (defined in the calculations below)

  # if resize=True the saved video will have his size reduced ONLY IF its width is bigger than max_width
  if (width > max_width):
    # we need to make width and height proportionals (to keep the proportion of the original video) so the image doesn't look stretched
    proportion = width / height
    # to do it we need to calculate the proportion (width/height) and we'll use this value to calculate the new height
    video_width = max_width
    video_height = int(video_width / proportion)
  else:
    video_width = width
    video_height = height

  return video_width, video_height