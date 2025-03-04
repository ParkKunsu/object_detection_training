import roboflow

# roboflow.login()

rf = roboflow.Roboflow(api_key="")

project = rf.workspace("team-roboflow").project("coco-128")
dataset = project.version(2).download("coco")
