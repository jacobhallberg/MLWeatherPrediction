from glob import glob
import os

from os.path import isfile, join
from shutil import copy
from tqdm import tqdm

import subprocess 
import re

filenames = [file for file in tqdm(os.listdir("../data")) if isfile(join('../data/', file)) and re.match(r'.*2014[0-9]{6}(0[0-9]|5[0-9])', file) and file[0:3] == "DAA"]

found_times = set()
fixed_filenames = []

for filename in tqdm(filenames):
    timestamp = filename[4:]
    if(timestamp[-2] == "5"):
        timestamp = timestamp[:-4] + str((int(timestamp[-4:-2]) + 1) % 24).zfill(2) + timestamp[-2:]

    if timestamp[:-2] not in found_times:
        fixed_filenames.append(filename)
        found_times.add(timestamp[:-2])

things_we_want = ["DTA", "N1U", "N0Z", "N0S"]
subprocess.call(["mkdir", "../output/{}/".format("DAA")])

for thing in things_we_want:
    subprocess.call(["mkdir", "../output/{}/".format(thing)])
count = 0 
for filename in tqdm(fixed_filenames):
    copy('../data/{}'.format(filename), "../output/{}/".format("DAA"))
    for thing in things_we_want:
        new_filename = filename.replace("DAA", thing)
        try:
            copy('../data/{}'.format(new_filename), "../output/{}/".format(thing))
        except:
            count += 1
            print("ERROR: {} was not found".format(new_filename))

print("Num of misaligned files: {}".format(count))
# hour = 0
# file_dict = {} 
# for hour in range(0,24):
#     for filename in filenames:
#         if filename[-4:-2] == str(hour).zfill(2):
#             print(filename)
#             copy('../'+filename, '../output/')
#             hour = (hour + 1)%24

# things_we_want = ["DSD", "DTA", "DAA", "OHA", "N0S"]
#Storm total precip dif, Storm total precip, one-hour-precip 1 level, one-hour-precip 16 level, Storm Relative Velocity, 

# """
# tk___10101014
# filename__tk___10101013
# filename__tk___10101012
# filename__tk___10101011

# """

# for filename in filenames:
#     #if match:
#     for thing in things_we_want:
#         new_filename = filename.replace(filename[:], thing)
#         #mv filename to dir

