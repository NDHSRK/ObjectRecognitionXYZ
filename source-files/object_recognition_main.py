from SampleRecognition import SampleRecognition
import argparse
import cv2
import os

def main():
    # Construct the argument parser and parse the arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str)  # e.g. \files\images\blue_sample.png
    ap.add_argument("--alliance", type=str)
    args = vars(ap.parse_args())

    ##TODO The image directory is os.getcwd() + "\files\images\"
    # You'll need this if you want to write output files.

    # load the image
    #current_working_directory = os.getcwd()
    image_full_path = os.getcwd() + args["image"]
    src = cv2.imread(image_full_path)
    if src is None:
        print('File not found')
        return

    ##TODO Currently this project uses a full image. To
    # support an ROI you have to port code from Java.
    # See C:\Users\lonep\OneDrive\Documents\FTC\FTC 2026 Post Season\CalculateXYZ\PortToPython.java

    # This project has a generic name, ObjectRecognitionXYZ,
    # and is intended to be merged into the project
    # CalculateXYZ as a replacement for the object recognition
    # that is part of Paco Garcia's open source project. But
    # the current project makes use of the work done on the
    # recognition of red and blue "samples" from the 2025 FTC
    # IntoTheDeep game.

    alliance = args["alliance"]
    alliance_instance = SampleRecognition.Alliance[alliance]

    ##TODO See error handling in
    ## 4/3/2026 main program section of FtcIntoTheDeepLimelight - with error handling
    # in file detect_sample_as_runPipeline.py

    recognition = SampleRecognition(alliance_instance.value)
    ret_val = recognition.perform_recognition(src)
    print(ret_val.status)

    if ret_val.status != SampleRecognition.SampleRecognitionReturn.RecognitionStatus.FAILURE:
        for one_object in ret_val.recognized_objects:
            # Print out the coordinates of each object's centerpoint
            # and its FTC angle.
            print("Recognized object with center x " + str(one_object.center_x) +
            ", center y " + str(one_object.center_y) +
            ", ftc angle " + str(one_object.ftc_angle))

if __name__ == "__main__":
    main()