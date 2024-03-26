import os
import cv2
import numpy as np

count=0
for frame in os.listdir("Frames"):
    frame_path = "Frames" + frame
    image = cv2.imread(frame_path)
    image = image[60:image.shape[0]-50,15:,:]

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numbers=[]
    image_copy = image.copy()
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        # Filter out small contours
        if area > 50:
            # Get bounding box coordinates of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small numbers and not numbers that were detected
            if 10 < w < 200 and 30 < h:
                #save numbers
                numbers.append((x,y,w,h))
                # Optional- Draw bounding box on the original image
                cv2.rectangle(binary_img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    
    for number in numbers:
        count+=1
        probabilities = []
        x,y,w,h = number
        number_pic = image_copy[y:y+h,x:x+w]

        number_pic = cv2.cvtColor(number_pic, cv2.COLOR_BGR2GRAY)    
        number_pic = cv2.resize(number_pic, (256,256),interpolation= cv2.INTER_NEAREST)
        for num in range(10):
            data = cv2.imread(os.path.join(f"NumberRecognition/Digits/data/bin_{num}.png"), cv2.IMREAD_GRAYSCALE)
            probabilities.append(np.sum(data==number_pic)/(data.shape[0]*data.shape[1])) 
        # if max(probabilities) < 0.8:
        #     prob,pred= max(probabilities),probabilities.index(max(probabilities))
        #     cv2.imwrite(f"NumberRecognition/Digits/data/10/{count}.png",number_pic)
