import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
class YOLO():
    def __init__(self):
        # TODO
        self.confThreshold = 0.5  
        self.nmsThreshold = 0.4
        self.inpWidth = 608
        self.inpHeight = 608
        classesFile = "coco.names"
        self.classes = None
        with open(classesFile,'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        modelConfiguration ="YoloV3.cfg"
        modelWeights ="yolov3.weights"
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def getOutputsNames(self):
    layersNames = self.net.getLayerNames()
    return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

YOLO.getOutputsNames = getOutputsNames

def drawPred(self, frame, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),8)
    label='%.2f' % conf
    if self.classes:
      assert(classId<len(self.classes))
      label='%s:%s' % (self.classes[classId],label)
    labelSize,baseLine=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), thickness=5)
    return frame
    
YOLO.drawPred = drawPred

def postprocess(self,frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
      for detection in out:
        scores=detection[5:]
        classId=np.argmax(scores)
        confidence=scores[classId]
        if confidence >self.confThreshold:
          centre_x=int(detection[0]*frameWidth)
          centre_y=int(detection[1]*frameHeight)
          width=int(detection[2]*frameWidth)
          height=int(detection[3]*frameHeight)
          left=int(centre_x- width/2)
          top=int(centre_y - height/2)
          classIds.append(classId)
          confidences.append(float(confidence))
          boxes.append([left,top,width,height])
    indices=cv2.dnn.NMSBoxes(boxes,confidences, self.confThreshold,self.nmsThreshold)
    for i in indices:
      i=i[0]
      box=boxes[i]
      left=box[0]
      top=box[1]
      width=box[2]
      height=box[3]
      outimage=self.drawPred(frame,classIds[i], confidences[i],left,top,left+width,top+height)
    return frame
    

YOLO.postprocess = postprocess

def inference(self,image):
    blob=cv2.dnn.blobFromImage(image,1/255,(self.inpWidth,self.inpHeight),[0,0,0],1,crop=False)
    self.net.setInput(blob)
    outs=self.net.forward(self.getOutputsNames())
    final_frame=self.postprocess(image,outs)
    return final_frame

YOLO.inference = inference
yolo = YOLO()

img =cv2.imread("static/Images/aarsh.jpeg")
oi = yolo.inference(img)
plt.figure(figsize=(15,15))
plt.imshow(oi[:,:,::-1])
plt.show()

"""### Detection on a video!

A video is just a set of frames, we will call the inference function for each frame of the video and save it.

"""

def videoInference(self,path):
  cap = cv2.VideoCapture(path)
  outputFile ='_yolo_out_py.avi'
  vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  while cv2.waitKey(1) < 0:
    ret,frame=cap.read()
    if not ret:
        print("Done processing !")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(25)
        cap.release()
        break
    blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
    self.net.setInput(blob)
    outs=self.net.forward(self.getOutputsNames())
    self.postprocess(frame,outs)
    t, _ = self.net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    vid_writer.write(frame.astype(np.uint8))
YOLO.videoInference=videoInference
yolo.videoInference("static/Images/LaneVideo.mp4")

# Commented out IPython magic to ensure Python compatibility.
from moviepy.editor import VideoFileClip
video_file = "static/Images/LaneVideo.mp4"
clip = VideoFileClip(video_file).subclip(0,1)
white_clip = clip.fl_image(yolo.inference)
# %time white_clip.write_videofile("movie.mp4",audio=False)

# import io
# import base64
# from IPython.display import HTML

# video = io.open('movie.mp4', 'r+b').read()
# encoded = base64.b64encode(video)
# HTML(data='''<video alt="test" controls width="320" height="240">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii')))

