import os
import number_recognition as nr

for frame in os.listdir("Frames"):
    frame_path = "Frames/" + frame
    print(frame)
    nr.main(frame_path)
    # results = pd.read_csv("/data/home/tal.dugma/Monitor-Recognition/results.csv") - Optional for saving/plotting results