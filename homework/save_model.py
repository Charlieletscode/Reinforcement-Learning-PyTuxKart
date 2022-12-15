from datetime import datetime
from os import path
import os
import shutil
try:
    time = datetime.now()
    time = str(time).split(".")[0].replace(
        "-", "_").replace(" ", "-").replace(":", "_")
    basePath = "./homework/"
    filePath = basePath + "model-" + str(time)
    if (not os.path.exists(filePath)):
        os.mkdir(filePath)
    src = path.join(basePath, 'planner.th')
    dst = path.join(filePath, 'planner.th')
    shutil.copy(src, dst)
    src = path.join(basePath, 'planner_best.th')
    dst = path.join(filePath, 'planner_best.th')
    shutil.copy(src, dst)
    src = path.join(basePath, 'planner.py')
    dst = path.join(filePath, 'planner.py')
    shutil.copy(src, dst)
    src = path.join(basePath, 'train_log.txt')
    dst = path.join(filePath, 'train_log.txt')
    shutil.copy(src, dst)
except:
    print("Cannot save model! Please finish your training.")
