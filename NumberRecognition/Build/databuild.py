import cv2
import os
import hashlib
import numpy as np
import shutil 
from number_recognition import find_numbers
# from videotoframes import extract_200_frames
from number_recognition import find_numbers
from number_recognition import main_color
#extract_200_frames(video_path="/home/tal/Desktop/Ms/Y1Sem1/ProjectYear1Sem1/code/videos/Monitor.mp4",output_folder="/home/tal/Desktop/Ms/Y1Sem1/ProjectYear1Sem1/code/MeasurementsRecognition")

def process_numbers_in_folder(folder_path, output_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    i = 0
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a valid image file (you may need to adjust this condition)
        if filename.lower().endswith(('.png')):
            # Get the full path of the image file
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            # Call your function find_numbers with the image path as parameter
            numbers = find_numbers(image)
            for number in numbers:
                i+=1
                x,y,w,h = number
                output_image_path = os.path.join(output_path, f"output_image{i}.png")
                number_image = image[y:y+h,x:x+w,:]
                cv2.imwrite(output_image_path, number_image.astype('uint8'))

#process_numbers_in_folder()
def sort_to_folders(folder_path):
    for filename in os.listdir(folder_path):
        # Check if the file is a valid image file (you may need to adjust this condition)
        if filename.lower().endswith(('.png')):
            # Get the full path of the image file
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            img_color = main_color(image)
            if img_color == 'E':
                pass
            classification = os.path.join(folder_path, img_color)
            os.rename(image_path,os.path.join(classification,filename))
#sort_to_folders()

def find_and_delete_duplicates(directory):
    # Dictionary to store hashes and corresponding file paths
    hash_dict = {}

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.png'):
            with open(filepath, 'rb') as f:
                # Calculate hash of the file's content
                file_hash = hashlib.md5(f.read()).hexdigest()

            # Check if hash already exists
            if file_hash in hash_dict:
                print(f"Duplicate found: {filepath}")
                # Remove duplicate file
                os.remove(filepath)
            else:
                # Add hash to dictionary
                hash_dict[file_hash] = filepath

    print("Duplicate deletion process completed.")
#find_and_delete_duplicates()

#img30 = cv2.imread("/home/tal/Desktop/Ms/Y1Sem1/ProjectYear1Sem1/code/Digits/temp copy/3.png")
#img32 = cv2.imread("/home/tal/Desktop/Ms/Y1Sem1/ProjectYear1Sem1/code/Digits/co2 copy/3.png")
#gray_img49 = cv2.cvtColor(img30, cv2.COLOR_BGR2GRAY)    
#gray_img157 = cv2.cvtColor(img32, cv2.COLOR_BGR2GRAY)
#_, binary_image50 = cv2.threshold(gray_img49, 50, 255, cv2.THRESH_BINARY)
#_, binary_image157 = cv2.threshold(gray_img157, 50, 255, cv2.THRESH_BINARY)
#print(binary_image50.shape,binary_image157.shape,)
#print(np.sum(binary_image157!=binary_image50))

def sort_images_by_shape(directory):
    # Create a dictionary to store image shapes and corresponding directories
    shape_directories = {}

    # List all files in the directory
    files = os.listdir(directory)

    # Iterate through each file
    for file in files:
        if file.endswith('.png'):
            # Get the full path of the file
            file_path = os.path.join(directory, file)

            # Open the image using OpenCV and get its shape
            img = cv2.imread(file_path)
            shape = img.shape[:2]  # OpenCV returns (height, width)

            # Convert the shape to a tuple
            shape_tuple = tuple(shape)

            # Check if the shape already exists in the dictionary
            if shape_tuple in shape_directories:
                # Move the file to the corresponding directory
                destination_directory = shape_directories[shape_tuple]
                shutil.move(file_path, destination_directory)
            else:
                # Create a new directory for the shape
                new_directory = os.path.join(directory, f"{shape[1]}x{shape[0]}")
                os.makedirs(new_directory, exist_ok=True)

                # Move the file to the new directory
                shutil.move(file_path, new_directory)

                # Update the dictionary
                shape_directories[shape_tuple] = new_directory

    print("Images sorted successfully.")
    
def binary_data(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
        output_path = "insert"
        output_image_path = os.path.join(output_path, f"bin_{file}")
        cv2.imwrite(output_image_path, bin_image)

def find_largest_image_shape(folder_path):
    files = os.listdir(folder_path)
    max_area = 0
    max_area_file = ""
    for file in files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        area = image.shape[0]*image.shape[1]
        if area > max_area:
            max_area = area
            max_area_file = file
    return(cv2.imread(os.path.join(folder_path, max_area_file)).shape)

def resize_images(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (256,256),interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(image_path, resized_image)
