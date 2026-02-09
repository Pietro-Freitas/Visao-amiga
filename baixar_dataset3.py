from roboflow import Roboflow

rf = Roboflow(api_key="xq7RkVszSKkU7bCtiiCv")
project = rf.workspace("ken-watanabe-mxlor").project("visao-amiga")
dataset = project.version(2).download("yolov8")

print("Dataset salvo em:", dataset.location)