# FILE MEANT TO WRITE THE RESULTS OF THE MEASUREMENTS
# TO THE DESIRE CSV FILE EVERY SECOND

import pandas as pd
def main(resutlts_path):
    resultdf = pd.DataFrame({'ECG':[],'spo2':[],'+co2':[],'NIBP':[],'temp':[]})
    with open(resutlts_path) as resultfile:
        category = ""
        for line in resultfile:
            if "ECG" in line:
                category = "ECG"
                length=0
            elif "spo2" in line:
                category = "spo2"
                length=0

            elif "+co2" in line:
                category = "+co2"
                length=0
            elif "NIBP" in line:
                category = "NIBP"
                length=0
            elif "temp" in line:
                category = "temp"
                length=0
            if category not in line:
                if "," in line:
                    numbers = line[:-1].split(",")
                else:
                    numbers = str(line[:-1])
                if type(numbers) == list:
                    for i in numbers:   
                        resultdf.at[length,category] = i
                        length+=1

                else:
                    resultdf.at[length,category] = numbers
                    length+=1
    resultdf.to_csv("results.csv",index=False)
