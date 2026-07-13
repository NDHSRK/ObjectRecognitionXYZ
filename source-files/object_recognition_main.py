from SampleRecognition import ImageParameters
from SampleRecognition import SampleRecognition
from ImageUtils import ImageUtils
import argparse
import cv2
import os
import traceback

from datetime import datetime


def main():
    # Construct the argument parser and parse the arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str)  # e.g. \files\images\blue_sample.png
    ap.add_argument("--alliance", type=str)
    args = vars(ap.parse_args())

    # load the image
    ## Changed in the Raspberry Pi version on 6/25/24;
    IMAGE_SUBDIRECTORY = os.sep + "files" + os.sep + "images" + os.sep  # const
    image_directory = os.getcwd() + IMAGE_SUBDIRECTORY
    image_filename = args["image"]
    image_full_path = image_directory + image_filename
    src = cv2.imread(image_full_path)
    if src is None:
        print('File not found')
        return

    # For writing output image files with a timestamp in
    # the filename.
    # Format: YYYY-MM-DD HH:MM:SS
    now = datetime.now()
    formatted_timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_file_preamble = ImageUtils.create_output_file_preamble(image_directory, image_filename, formatted_timestamp)

    # This project has a generic name, ObjectRecognitionXYZ,
    # and is intended to be merged into the project
    # CalculateXYZ as a replacement for the object recognition
    # that is part of Paco Garcia's open source project. But
    # the current project makes use of the work done on the
    # recognition of red and blue "samples" from the 2025 FTC
    # IntoTheDeep game.

    alliance = args["alliance"]
    alliance_instance = SampleRecognition.Alliance[alliance]

    # Currently this project uses a full image.
    # To support an ROI you use a Python version of Java
    # VisionParameters.ImageParameters, filled in with
    # the ROI values x origin y origin, width, height.
    # Pass an instance of the ImageParameters class to
    # the function perform_recognition.
    image_parameters = ImageParameters(src, 640, 480, 0, 0, 640, 480)

    try:
        recognition = SampleRecognition(alliance_instance.value, output_file_preamble)
        ret_val = recognition.perform_recognition(src, image_parameters)
        print(ret_val.status)

        # When you introduce an ROI you have to ensure that
        # the final coordinates of the recognized object are
        # relative to the full image, not just the ROI - because
        # CalculateXYZ needs the full-image coordinates. See enhanced
        # logging in c++ CLOpenCVTestbed4.NineDotContours and
        # ShapeDrawing.
        if ret_val.status != SampleRecognition.SampleRecognitionReturn.RecognitionStatus.FAILURE:
            for one_object in ret_val.recognized_objects:
                # Print out the coordinates of each object's centerpoint
                # and its FTC angle.
                if image_parameters.roi_x != 0 or image_parameters.roi_y != 0:
                    print("Object location in the ROI with center at x " + str(one_object.center_x) + ", y " + str(
                        one_object.center_y))

                print("Object location in the full image with center at x " + str(
                    one_object.center_x + image_parameters.roi_x) + ", y " +
                      str(one_object.center_y + image_parameters.roi_y) +
                      ", ftc angle " + str(one_object.ftc_angle))

    except Exception as e:
        print(repr(e))

        # Extracts all trace entries
        summary = traceback.extract_tb(e.__traceback__)

        # Get the very last frame where the crash actually happened
        last_frame = summary[-1]

        print(f"File: {last_frame.filename}")
        print(f"Function: {last_frame.name}")
        print(f"Line Number: {last_frame.lineno}")
        print(f"Code line: {last_frame.line}")

if __name__ == "__main__":
    main()
