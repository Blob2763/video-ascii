import cv2
import os
import shutil

from PIL import Image, ImageFont, ImageDraw, ImageColor

import numpy as np

from config import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the FPS (frames per second) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Release the video capture object
    cap.release()

    print(f"Video fps: {fps}")

    return fps


def video_to_frames(video_path):
    # Directory to save the images
    output_directory = "frames/"

    # Create the output directory if it doesn"t exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize a frame counter
    frame_counter = 0

    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we"ve reached the end of the video
        if not ret:
            break

        # Save the frame as an image
        image_filename = os.path.join(output_directory, f"frame_{frame_counter:04d}.jpg")
        cv2.imwrite(image_filename, frame)

        # Increment the frame counter
        frame_counter += 1

    # Release the video capture object
    cap.release()

    # Print a message when the process is complete
    print(f"Extracted {frame_counter} frames as images to {output_directory}")


def rgb_to_brightness(pixel):
    # Extract the individual color channels
    r = pixel[0]
    g = pixel[1]
    b = pixel[2]

    # Calculate brightness using the formula
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    return brightness


def get_font_dimensions():
    # Load the font
    font = ImageFont.truetype(FONT, FONT_SIZE)

    # Get the font height (ascent + descent)
    font_dimensions = font.getsize(CHARS[0])

    return font_dimensions


def image_to_ascii(image_path):
    chars_len = len(CHARS)
    image = Image.open(image_path)
    pixels = np.array(image)
    image.close()

    font_width, font_height = get_font_dimensions()
    font_aspect_ratio = font_height / font_width

    image_ascii = ""

    scaled_rows = int(np.floor(len(pixels) / SCALE / font_aspect_ratio))

    for rowIdx in range(scaled_rows):
        row = pixels[int(np.floor(rowIdx * SCALE * font_aspect_ratio))]

        row_ascii = ""

        scaled_columns = int(np.floor(len(row) / SCALE))
        for pixelIdx in range(scaled_columns):
            pixel = row[pixelIdx * SCALE]

            brightness = rgb_to_brightness(pixel)
            normalized_brightness = np.floor((brightness / 256) * chars_len)
            char = CHARS[int(normalized_brightness)]

            row_ascii += char

        image_ascii += row_ascii + "\n"

    return image_ascii


def text_to_image(
    text: str,
    font_filepath: str,
    font_size: int,
    color: (int, int, int)
):

    font = ImageFont.truetype(font_filepath, size=font_size)
    box = font.getsize_multiline(text)
    img = Image.new("RGB", (box[0], box[1]), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_point = (0, 0)
    draw.multiline_text(draw_point, text, font=font, fill=color, align="left")

    return img


def frames_to_ascii():
    frames = os.listdir("frames/")
    frame_count = len(frames)

    # Directory to save the images
    output_directory = "ascii/"

    # Create the output directory if it doesn"t exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through the video frames
    for frame_idx in range(frame_count):
        frame_path = f"frames/frame_{frame_idx:04d}.jpg"
        ascii_frame = image_to_ascii(frame_path)
        ascii_image = text_to_image(ascii_frame, FONT, FONT_SIZE, (255, 255, 255))

        # Save the frame as an image
        image_filename = os.path.join(
            output_directory, f"ascii_{frame_idx:04d}.jpg")
        ascii_image.save(image_filename)

    # Print a message when the process is complete
    print(f"Converted {frame_count} frames to ASCII in {output_directory}")


def ascii_frames_to_video(output_video):
    # Directory containing the JPG images
    image_directory = "ascii/"

    # Get the list of image files in the directory
    image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith(".jpg")]

    # Sort the image files by filename (assuming they are in sequential order)
    image_files.sort()

    # Get the first image to determine the frame size
    first_image = cv2.imread(image_files[0])
    frame_height, frame_width, _ = first_image.shape

    # Define the codec and create a VideoWriter object
    fps = get_fps(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Loop through the image files and add them to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Video saved as {output_video}")


def delete_directories():
    # Remove the Directory
    shutil.rmtree(os.path.join(PARENT, "ascii"))
    shutil.rmtree(os.path.join(PARENT, "frames"))


print("Extracting frames")
video_to_frames(VIDEO_PATH)
print("Converting frames to ASCII")
print(f"Font dimensions: {get_font_dimensions()}")
frames_to_ascii()
print("Making video")
ascii_frames_to_video(OUTPUT_PATH)
print("Finishing off...")
delete_directories()
print("Done!")
