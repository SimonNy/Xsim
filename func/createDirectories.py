""" Create directories to save the relevant files for a given file """
import os
import csv
def createDirectories(filename, folder, const_names, const_vals):
    dirName = filename
    try:
        # Create target Directory
        os.mkdir(folder+filename)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")

    dirName = 'subframes'
    try:
        # Create target Directory
        os.mkdir(folder+filename+'/'+dirName)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")
    # Create directory
    dirName = 'CCDreads'
    try:
        # Create target Directory
        os.mkdir(folder+filename+'/'+dirName)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")

    with open(folder+filename+'/''details.csv', 'w', ) as myfile:
        wr = csv.writer(myfile)
        # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        test = zip(const_names, const_vals)
        for row in test:
           wr.writerow([row])



