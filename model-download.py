from roboflow import Roboflow
rf = Roboflow(api_key="TD5xEPznLTfQhznZz4pf")
project = rf.workspace("rp-project").project("circuit-recognition")
version = project.version(9)
dataset = version.download("yolov8")
                