import cv2
import pytesseract
# Load image- in case we want to run tesseract only once. (saves runtime by a factor of 2)
image = cv2.imread("/data/home/tal.dugma/Monitor-Recognition/MNEW/MeasurementsRecognition/frames2/frame_159594.png")
image = image[60:image.shape[0]-50,15:,:]
names = image[:,image.shape[1]-475:image.shape[1]-425,:]

# Convert the image to grayscale
gray_image = cv2.cvtColor(names, cv2.COLOR_BGR2GRAY)

# Use Tesseract OCR to recognize numbers
custom_config = r'--oem 3 --psm 6 outputbase letters'
result = pytesseract.image_to_data(gray_image, config=custom_config,output_type=pytesseract.Output.DICT)
measurements_keys = {"ECG":[],"spo2":[],"+co2":[],"NIBP":[]}
for i in range(len(result['text'])):
    # Check if the word is not empty and has valid bounding box coordinates
    if result['text'][i].strip() != '' and int(result['conf'][i]) > -1:
        # Extract word and bounding box coordinates
        word = result['text'][i]
        if word in {"ECG","spo2","+co2","NIBP"}:
            x = result['left'][i]
            y = result['top'][i]
            width = result['width'][i]
            height = result['height'][i]
            measurements_keys[word].append((x,y))
            #OPTIONAL: Draw a rectangle around the word on the original image
            cv2.rectangle(names, (x, y), (x + width, y + height), (0, 255, 0), 2)
#update locations for original image
for key in measurements_keys:
    for i in range(len(measurements_keys[key])):
        measurements_keys[key][i] = (measurements_keys[key][i][0]+image.shape[1]-475,measurements_keys[key][i][1])
# print(measurements_keys)        
# Display the image with bounding boxes
# cv2.imshow('Image with Bounding Boxes', names)
# cv2.waitKey(0)
# cv2.destroyAllWindows()