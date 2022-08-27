import os
import sys
from itertools import combinations
from random import sample
import random
import csv

NUM_SAMPLES = 100
ROOT_NAME1 = "OLI"
ROOT_NAME2 = "OLIVINE"
NUM1 = 11
NUM2 = 49

HEADERS = ["Image1", "Image2", "Label"]

def change_names(path, root_name):
    i = 0
    for filename in os.listdir(path):
        my_dest = f"{root_name}_" + str(i) + ".jpg"
        my_source = path + filename
        my_dest = path + my_dest
        os.rename(my_source, my_dest)
        i += 1

def create_pos_neg_csv():
    num1 = NUM1
    num2 = NUM2

    with open("dataset.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        pairs1 = sample(list(combinations(range(num1), 2)), NUM_SAMPLES//2)
        for pair in pairs1:
            str1 = f"{ROOT_NAME1}_{pair[0]}.jpg"
            str2 = f"{ROOT_NAME1}_{pair[1]}.jpg"
            writer.writerow([str1, str2, str(1)])

        pairs2 = sample(list(combinations(range(num2), 2)), NUM_SAMPLES)
        for pair in pairs2:
            str1 = f"{ROOT_NAME2}_{pair[0]}.jpg"
            str2 = f"{ROOT_NAME2}_{pair[1]}.jpg"
            writer.writerow([str1, str2, str(1)])

        for i in range(2*NUM_SAMPLES):
            str1 = f"{ROOT_NAME1}_{random.randint(0,num1-1)}.jpg"
            str2 = f"{ROOT_NAME2}_{random.randint(0,num2-1)}.jpg"
            writer.writerow([str1, str2, str(0)])



if __name__ == "__main__":
    create_pos_neg_csv()
