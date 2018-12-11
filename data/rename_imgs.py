from glob import glob
import subprocess
from os import listdir, getcwd


for upper in glob("../data/*"):
    for filename in glob("{}/*".format(upper)):
        code = filename[71:74]
        new_filename = code + "_" + filename[-12:]
        subprocess.call(["mv", filename, "../data/{}".format(new_filename)])

