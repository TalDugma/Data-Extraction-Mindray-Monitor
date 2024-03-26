import cv2
import os 
# from number_recognition import recognize_number
# from number_recognition import find_numbers
import numpy as np

# video_path = "" - insert video path

def get_total_frames(video): 
    """
    Function gets the total number of frames of a given video
    
    Parameters:
    -video (after the path was captured)

    Returns:
    Error message if failed
    total_frames (int) if succeded       
    """

    # Check if the video file is opened successfully
    if not video.isOpened():
        # Prints the video path that couldn't be opened 
        print(f"Error opening video file: {video.get(cv2.CAP_PROP_POS_FRAMES)}")
        return None

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return total_frames

def extract_frames(video_path,output_folder): 
    """
    Function extracts frames from a video and saves in user's desired path
    
    Parameters:
    -video_path
    -output_folder

    Returns:
    Error message if failed, and the frame number that caused the error      
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the frames and save them as images
    frame_count = 0
    total_frames = get_total_frames(cap)
    if total_frames is None:
        return False
    ret = True
    print("total frames: ",total_frames)
    while (frame_count < total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_count}. Exiting.")
            return False
        # Save the frame as an image
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    #Release the video capture object
    cap.release()
    return True

def extract_frames_starting_minute(video_path, output_folder):
    """
    Function extracts frames from a video and saves in user's desired path, starting from where the minute changes
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video file has no frames.")
        return False

    # Set the frame position to start reading from halfway
    starting_point = total_frames // 5
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_point)

    # Set the frame interval to save
    frame_interval = 60  # Save every 60th frame
    frame_count = starting_point

    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {frame_count}. Exiting.")
        return False    
    timer = frame[20:60,-100:]
    #convert to binary
    timer = cv2.cvtColor(timer, cv2.COLOR_BGR2GRAY)
    _,timer = cv2.threshold(timer, 200, 255, cv2.THRESH_BINARY)
    # Read frames and save every 1th frame
    frame_number = starting_point
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_count}. Exiting.")
            return False
        next_timer = frame[20:60,-100:]
        next_timer = cv2.cvtColor(next_timer, cv2.COLOR_BGR2GRAY)
        _,next_timer = cv2.threshold(next_timer, 200, 255, cv2.THRESH_BINARY)

        #timer changed

        if mse(next_timer,timer) > 70:
            print(frame_count-frame_number)
            frame_number = frame_count
            timer = next_timer
            # Save the frame as an image

            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)

        # Move to the next frame position
        frame_count += frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # Release the video capture object
    cap.release()
    return True

def mse(imageA, imageB):
    # Compute the mean squared error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def extract_unique_frames(video_path, output_folder):

    """
    Function extracts unique frames from a video and saves in user's desired path
    NOTE: maybe I should add a parameter for the threshold of the mse
    NOTE: maybe I should calculate the mse between the frames without the clock, since the clock is changing every minute
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    previous_frame = None
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video file has no frames.")
        return False

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_count}. Exiting.")
            return False

        if previous_frame is not None:
            # Check if the current frame is different from the previous frame
            if mse(frame, previous_frame) > 5:  
                # Save the frame as an image if it is different from the previous frame
                frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
                cv2.imwrite(frame_path, frame)
                previous_frame = frame
            
        else:
            # Save the first frame as an image
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            previous_frame = frame

        # Move to the next frame position
        frame_count += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # Release the video capture object
    cap.release()
    return True

def check_frame_hypothesis(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)


    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video file has no frames.")
        return False

    # Set the frame position to start reading from halfway
    starting_point = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_point)

    # Set the frame interval to save
    frame_count = starting_point

    ret, frame = cap.read()
    if not ret:
        print(f"Error reading frame {frame_count}. Exiting.")
        return False    
    timer = frame[20:60,-100:]
    #convert to binary
    timer = cv2.cvtColor(timer, cv2.COLOR_BGR2GRAY)
    _,timer = cv2.threshold(timer, 200, 255, cv2.THRESH_BINARY)
    # Read frames and save every 1th frame
    frame_number = starting_point
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_count}. Exiting.")
            return False
        next_timer = frame[20:60,-100:]
        next_timer = cv2.cvtColor(next_timer, cv2.COLOR_BGR2GRAY)
        _,next_timer = cv2.threshold(next_timer, 200, 255, cv2.THRESH_BINARY)

        #timer changed

        if mse(next_timer,timer) > 70:
            if frame_count-frame_number != 3600:
                #show pictures
                cv2.imwrite(f"frame_{frame_count}.png", frame)
                cv2.imwrite(f"timer_{frame_count}.png", timer)
                cv2.imwrite(f"next_timer_{frame_count}.png", next_timer)
                print(f"HYPOTHESIS FAILED-{frame_count-frame_number}")
            frame_number = frame_count
            timer = next_timer


        # Move to the next frame position
        frame_count += 60
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # Release the video capture object
    cap.release()
    return True

