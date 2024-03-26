import cv2
import numpy as np
import os
import pytesseract
from collections import defaultdict 
from pytesseract_measure import measurements_keys
import time
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models, transforms

#OPTIONAL: Load the pre-trained VGG model for transfer learning
# Load the pre-trained VGG model
# vgg = models.vgg16(pretrained=True)
# # Remove the last layer
# features = list(vgg.classifier.children())[:-1]
# # Add the new layer
# features.extend([nn.Linear(4096, 11)])
# # Replace the model's classifier
# vgg.classifier = nn.Sequential(*features)
# # Load the state dictionary
# vgg.load_state_dict(torch.load('transfer_model.pth'))
# print("Model loaded")

def warning_or_critical(image):
    """
    function that checks if the image is a warning or critical image. I chose to decide if the image is
    warning or critical by the pixels in the 4 corners of the image.
    param: numpy 3D array image- read with cv2- an image of an object detected in the frame
    returns "w" for warning, "c" for critical and "n" for normal 
    """
    corners = [image[3,3],image[3,image.shape[1]-3],image[image.shape[0]-3,3],image[image.shape[0]-3,image.shape[1]-3]]
    #critical for all corners to be red, warning for all corners to be yellow, normal otherwise
    corner_colors = []
    for corner in corners:
        if corner[2]>150 and corner[1]<100 and corner[0]<100:
            corner_colors.append("c")
        if corner[2]>150 and corner[1]>150 and corner[0]<100:
            corner_colors.append("w")
    if corner_colors == ["c","c","c","c"]:
        return "c"  
    if corner_colors == ["w","w","w","w"]:
        return "w"
    return "n"

def k_means(datapoints,k):
    """
    param: list datapoints- (x,y) coordinates of the datapoints

    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(datapoints)
    return kmeans.cluster_centers_

def measurenent_location(image):
    """
    function gets the image and returns the location of the measurements (except of temp) in the image
    """
    names = image[60:-100,image.shape[1]-475:image.shape[1]-425,:]
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

def main_color(image):
    """
    param: numpy 3D array image- read with cv2- an image of an object detected in the frame
    returns string main_color- the color that has the most pixels in the image
    """
    gbryw_count = {"ECG":0, "temp":0, "NIBP":0, "spo2":0, "+co2":0}
    for y in range(image.shape[0]):  # Iterate over rows (y-coordinate)
        for x in range(image.shape[1]):  # Iterate over columns (x-coordinate)
            blue, green, red = image[y, x]  # Get BGR color at this pixel
            
            # if a pixel isn't black- It's the color that we are looking for
            if blue > 50 or green > 50 or red > 50:
                # green
                if blue < 50 and green > 150 and red < 50:
                    gbryw_count["ECG"] += 1
                # blue
                if blue > 100 and green < 50 and red < 50:
                    gbryw_count["temp"] += 1
                # red
                elif blue < 50 and green < 50 and red > 150:
                    gbryw_count["NIBP"] += 1
                # yellow
                elif blue < 50 and green > 150 and red > 150:
                    gbryw_count["spo2"] += 1
                # white
                elif blue > 200 and green > 200 and red > 200:
                    gbryw_count["+co2"] += 1
    # return the color with the most pixels
    max = 0
    
    for color in gbryw_count:
        if gbryw_count[color] > max:
            max = gbryw_count[color]
            main_color = color
    if max > 15:
        return main_color        
    # return 'E' if couldn't find main color- final filter of bounding boxes
    return "E"

def find_numbers(image,min_area_threshold = 50 , inverse = False):
    """
    param: image (read by path with cv2)
    finds numbers in frame and bounds them in boxes. shows the frame with the bounding boxes
    returns list numbers- a list of the numbers' bounding boxes' indexes 
    """
    

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert binary the image- In case of a warning or critical number
    if inverse:
        _,binary_img = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Threshold the grayscale image to obtain a binary mask
    else:
        _, binary_img = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #List of contours that represent our desired numbers
    numbers=[]
    image_copy = image.copy()


    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        # Filter out small contours
        if area > min_area_threshold:
            # Get bounding box coordinates of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small numbers and not numbers that were detected
            if 10 < w < 200 and 30 < h:
                #save numbers
                numbers.append((x,y,w,h))
                # Optional- Draw bounding box on the original image
                cv2.rectangle(binary_img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    # Display the image with bounding boxes
    cv2.imwrite('Image with Bounding Boxes.png', binary_img)
    return numbers

def classify_number(number,image):
    """
    param: numpy 3D array image- read with cv2,  number- numbers' bounding boxe 
    returns string measure- the measurement class of the number 
    """

    x,y,w,h = number 
    original_number = image[y:y+h,x:x+w,:]

    state = warning_or_critical(original_number)
    if state == 'c':
        return "alertC", original_number, (x,y)
    elif state == 'w':
        return "alertW", original_number, (x,y)
    else: 
        return main_color(original_number),original_number,(x,y)   
    
def measure_dict(image):
    """ 
    param: numpy 3D array image- read with cv2
    returns dictionary measurements- a dictionary of the measurements and their numbers' bounding boxes' indexes
    """
    numbers = find_numbers(image)
    measurements = {'alertW':[],'alertC':[], 'ECG':[],'spo2':[],'+co2':[],'NIBP':[],'temp':[]}
    for number in numbers:
        measure,original_number,location = classify_number(number, image)
        if measure=='alertC':
            #convert red background to black
            for y in range(original_number.shape[0]):
                for x in range(original_number.shape[1]):
                    if original_number[y,x,2]>150 and original_number[y,x,1]<150 and original_number[y,x,0]<150:
                        original_number[y,x] = [0,0,0]
            # cv2.imwrite("alertC.png",original_number)
            
            #split to digits

            digits = find_numbers(original_number, min_area_threshold=0,inverse = False)
            for digit in digits:
                (x,y) = tuple(x + y for x, y in zip(location, (digit[0],digit[1])))
                w,h = digit[2],digit[3]
                measurements['alertC'].append((image[y:y+h,x:x+w,:],(x,y)))
        elif measure=='alertW':
            #split to digits
            digits = find_numbers(original_number, min_area_threshold=0,inverse = True)
            for digit in digits:
                (x,y) = tuple(x + y for x, y in zip(location, (digit[0],digit[1])))
                w,h = digit[2],digit[3]
                measurements['alertW'].append((image[y:y+h,x:x+w,:],(x,y)))
        elif measure!="E": 
            measurements[measure].append((original_number,location))
        else:
            print("Couldn't classify number in location: ",location)
            cv2.imshow("Couldn't classify",original_number)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()     
    return measurements        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)  # Adjusted input size
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(120, 11)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)  # Adjusted size
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        return x
    
def recognize_number(number, inverse = False):
    """
    param: numpy 3D array number- read with cv2
    returns tuple (probability,tag)- the probability of the number being a certain tag and the tag
    """
    probabilities = []
    number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
    if inverse:
        _, number = cv2.threshold(number, 15, 255, cv2.THRESH_BINARY_INV)
    else:    
        _, number = cv2.threshold(number, 15, 255, cv2.THRESH_BINARY)
    number = cv2.resize(number, (256,256),interpolation= cv2.INTER_NEAREST)
    for num in range(10):
        data = cv2.imread(os.path.join(f"/data/home/tal.dugma/Monitor-Recognition/Digits/data/bin_{num}.png"), cv2.IMREAD_GRAYSCALE)
        probabilities.append(np.sum(data==number)/(data.shape[0]*data.shape[1])) 
    return (max(probabilities),probabilities.index(max(probabilities)))

def recognize_number_cnn(number, inverse = False,threshold = 0.8):
    model = Net()
    state_dict = torch.load('/data/home/tal.dugma/Monitor-Recognition/conv_model.pth')
    model.load_state_dict(state_dict)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
    if inverse:
        _, number = cv2.threshold(number, 15, 255, cv2.THRESH_BINARY_INV)
    else:
        _, number = cv2.threshold(number, 15, 255, cv2.THRESH_BINARY)
    number = cv2.resize(number, (256,256),interpolation= cv2.INTER_NEAREST)
    
    output = model(torch.tensor(number).unsqueeze(0).unsqueeze(0).float())
    #apply softmax
    output = F.softmax(output,dim=1)
        
    probabilities = output.detach().numpy()[0]    
    #move first element to the end
    probabilities = np.roll(probabilities,1)
    if max(probabilities) > threshold and probabilities.argmax() < 11:
        return (max(probabilities),probabilities.argmax())
    return (0,0)
def recognize_number_transfer_cnn(number, inverse = False,threshold = 0.8):
    
    
    device = torch.device('cpu')
    number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
    number = cv2.cvtColor(number, cv2.COLOR_GRAY2RGB)
    number = cv2.resize(number, (256, 256), interpolation=cv2.INTER_NEAREST)
    #save number as image
    cv2.imwrite("number.png",number)
    number = transforms.ToTensor()(number)
    output = vgg(number.unsqueeze(0))
    output = F.softmax(output, dim=1)
    probabilities = output.detach().numpy()[0]
    #move ellement in index 2 to the end
    probabilities = np.append(np.delete(probabilities, 2), probabilities[2])

    if max(probabilities) > threshold and probabilities.argmax() < 11:
        return (max(probabilities), probabilities.argmax()) 
    return (0,0)

def final_recognize(measurements,confidence = 0.82):
    """
    param: dictionary measurements- a dictionary of the measurements and their numbers' bounding boxes' indexes
    returns dictionary recognized- a dictionary of the recognized measurements that are probably numbers and their numbers' bounding boxes' indexes
    """
    recognized = {}
    for measure in measurements:
        recognized[measure] = []
        for number,location in measurements[measure]:
            if measure == "alertW":
                if recognize_number(number,inverse=True)[0] >= confidence:
                    recognized[measure].append(((number,(recognize_number(number,inverse = True))[1]),location))     
            else:
                rec = recognize_number(number)
                max_prob,pred = rec[0],rec[1]
                if measure == "alertC":
                    if max_prob >= confidence:
                        recognized[measure].append(((number,pred),location))
                elif max_prob >= confidence:
                    recognized[measure].append(((number,pred),location))
    return recognized

def custom_sort(item):
    # Round y to the nearest multiple of 10
    rounded_y = round(item[1][1] / 10) * 10
    return rounded_y, item[1][0]

def sort_recognized(recognized):
    for measure in recognized:
        recognized[measure].sort(key=custom_sort)
    return recognized

# get functions for each measurement: ecg, spo2, co2, nibp, temp
 
def get_ecg(ecg):   
    result=0
    if len(ecg)<1 or len(ecg)>3:
        return 0
    for ((number,tag),loctaion) in ecg:
        result*=10
        result+=tag
    return result
def get_spo2(spo2):
    spo2_1,spo2_2,spo2_3 = 0,0,0
    if len(spo2)==8:
        spo2_1 =  100*spo2[0][0][1]+10*spo2[1][0][1]+spo2[2][0][1]
        spo2_2 = spo2[3][0][1] + 0.1 * spo2[4][0][1] + 0.01 * spo2[5][0][1]
        spo2_2 = round(spo2_2, 2)
        spo2_2 = round(spo2_2, 2)
        spo2_3 =  spo2[6][0][1]*10 + spo2[7][0][1]
    elif len(spo2)==7:
        spo2_1 =  10*spo2[0][0][1]+spo2[1][0][1]
        spo2_2 =  spo2[2][0][1]+0.1*spo2[3][0][1]+0.01*spo2[4][0][1]
        spo2_2 = round(spo2_2, 2)
        spo2_2 = round(spo2_2, 2)
        spo2_3 =  spo2[5][0][1]*10 + spo2[6][0][1]    
    else:
        return -1,-1,-1
    return spo2_1,spo2_2,spo2_3
def get_co2(co2):
    co2_1,co2_2,co2_3 = 0,0,0
    if len(co2)!=5:
        return -1,-1,-1
    else:
        co2_1 =  10*co2[0][0][1]+co2[1][0][1]
        co2_2 =  10*co2[2][0][1]+co2[3][0][1]
        co2_3 =  co2[4][0][1]
    return co2_1,co2_2,co2_3
def get_nibp(nibp):
    nibp_1, nibp_2,= [],[]
    if len(nibp)>8:
        return -1,-1,-1
    else:
        y_low = nibp[0][1][1]
        for ((number,tag),location) in nibp:
            if location[1] - y_low < 20:
                nibp_1.append(tag)
            else:
                nibp_2.append(tag)
    if len(nibp_1)==4 and len(nibp_2)==3:
        return 10*nibp_1[0]+nibp_1[1],10*nibp_1[2]+nibp_1[3],10*nibp_2[0]+nibp_2[1]+nibp_2[2]
    if len(nibp_1)==4 and len(nibp_2)==2:
        return 10*nibp_1[0]+nibp_1[1],10*nibp_1[2]+nibp_1[3],10*nibp_2[0]+nibp_2[1]
    if len(nibp_1)==5 and len(nibp_2)==2:
        return 100*nibp_1[0]+10*nibp_1[1]+nibp_1[2],10*nibp_1[3]+nibp_1[4],10*nibp_2[0]+nibp_2[1]            
    if len(nibp_1)==5 and len(nibp_2)==3:
        return 100*nibp_1[0]+10*nibp_1[1]+nibp_1[2],10*nibp_1[3]+nibp_1[4],10*nibp_2[0]+nibp_2[1]+nibp_2[2]
    return -1,-1,-1
def get_temp(temp):
    if len(temp)!=3:
        return -1
    else:
        return 10*temp[0][0][1]+temp[1][0][1]+0.1*temp[2][0][1]

def get_results(measure,l,results):
    """
    replaces get_ecg, get_spo2, get_co2, get_nibp, get_temp and writes the results to the results file
    """
    if measure=="ECG":
        result = get_ecg(l)
        if result==0:
            results.write("Couldn't recognize ecg" + "\n")
            print("Couldn't recognize ecg")
        else:
            results.write(str(result) + "\n")
    if measure=="spo2":
        spo2_1,spo2_2,spo2_3 = get_spo2(l)
        if (spo2_1,spo2_2,spo2_3) == (-1,-1,-1):
            results.write("Couldn't recognize spo2" + "\n")
            print("Couldn't recognize spo2")
        else:    
            results.write(str(spo2_1) + "\n")
            results.write(f"{spo2_2}, {spo2_3}" + "\n")
    if measure=="+co2":
        co2_1,co2_2,co2_3 = get_co2(l)
        if (co2_1,co2_2,co2_3) == (-1,-1,-1):
            results.write("Couldn't recognize co2" + "\n")
            print("Couldn't recognize co2")
        else:    
            results.write(str(co2_1) + "\n")
            results.write(f"{co2_2}, {co2_3}" + "\n")
    if measure=="NIBP":
        nibp_1,nibp_2,nibp_3 = get_nibp(l)
        if (nibp_1,nibp_2,nibp_3) == (-1,-1,-1):
            results.write("Couldn't recognize nibp" + "\n")
            print("Couldn't recognize nibp")
        else:
            results.write(f"{nibp_1}, {nibp_2}" + "\n")
            results.write(str(nibp_3) + "\n")
    if measure=="temp":
        temp = get_temp(l)
        if temp == -1:
            results.write("Couldn't recognize temp" + "\n")
            print("Couldn't recognize temp")
        else:
            results.write(str(temp) + "\n")    

def sort_list(measure):
    """
    function takes the measures sorted by sort_recognized and sorts the numbers to mini-lists based on their y value
    """
    sorted_parts = {}

    for part in measure:
        rounded_y = round(part[1][1] / 10) * 10
        rounded_y_up = rounded_y+10
        rounded_y_down = rounded_y-10
        if rounded_y not in sorted_parts:
            if rounded_y_up not in sorted_parts:
                if rounded_y_down not in sorted_parts:
                    sorted_parts[rounded_y] = [part]
                else:
                    sorted_parts[rounded_y_down].append(part)
            else:
                sorted_parts[rounded_y_up].append(part)        
        else:
            sorted_parts[rounded_y].append(part)

    return sorted_parts
        
def recognize_alerted_measurements(reco_mini_lists,alert_type,image):
    measure_alerted = defaultdict(list)
    alerts = reco_mini_lists[alert_type]
    # measurements_locations = measurenent_location(image)
    measurements_locations = measurements_keys
    locations_ranges = []
    last_y = (measurements_locations['ECG'])[0][1]
    #get measurements y-ranges
    for measure in measurements_locations:
        if measure!='ECG':
            locations_ranges.append((last_y,measurements_locations[measure][0][1]))
            last_y = locations_ranges[-1][1]
    locations_ranges.append((last_y,image.shape[0]))       
    #
    for alert in alerts:
        x_alert = alert[1][0]
        y_alert = alert[1][1]
        if x_alert<measurements_locations["ECG"][0][1]:
            measure_alerted["temp"].append(alert)
        else:
            for i in range(len(locations_ranges)):
                if locations_ranges[i][0]<y_alert<locations_ranges[i][1]:
                    measure_alerted[list(measurements_locations.keys())[i]].append(alert)
    return measure_alerted

def main(image_path):
    results_path = "results.txt"
    start_time = time.time()
    image = cv2.imread(image_path)
    image = image[60:image.shape[0]-50,15:,:]
    measurements = measure_dict(image)
    recognized = final_recognize(measurements)  
    measure_alertedC = recognize_alerted_measurements(recognized,"alertC",image)
    measure_alertedW =  recognize_alerted_measurements(recognized,"alertW",image)
    #insert the alerts to the recognized dictionary
    for measure in measure_alertedC:
        recognized[measure] += measure_alertedC[measure]
    for measure in measure_alertedW:
        recognized[measure] += measure_alertedW[measure]
    #sort the recognized dictionary    
    recognized = sort_recognized(recognized) #- NOTE: currently not in use

    #results is a new file that stores the recognized measurements
    #if the file already exists, it will be deleted and a new one will be created
    if os.path.exists(results_path):
        os.remove(results_path)
    
    results = open(results_path,"w")
    results.write(f"Recognized measurements in {image_path}:\n")

    #write the recognized measurements to the file
    for measure in recognized:
        if measure == "alertC" or measure == "alertW":
            continue
        results.write(measure + ":\n")
        if measure in measure_alertedC.keys():
            results.write("Critical!\n")
        if measure in measure_alertedW.keys():
            results.write("Warning!\n")
        get_results(measure,recognized[measure],results)
    results.close()


    import ResultsToCSV
    ResultsToCSV.main(results_path)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the total runtime
    total_runtime = end_time - start_time
    
    # Print the total runtime
    print("Total runtime:", total_runtime, "seconds")

