from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.modules['Image'] = Image 


app = Flask(__name__)

# pathmodel = r"static/Trained_Model/Leaf_Detection.h5"
# cnn = load_model( pathmodel )

APP_ROOT= os.path.dirname(os.path.abspath(__file__))
target = os.path.join(APP_ROOT,'static/')
app.config["DEBUG"] = True

picFolder = os.path.join('static','User-Image')
#os.path.isfile('mydirectory/myfile.txt')  ---> to  check whether file existe or not
app.config["UPLOAD_FOLDER"] = picFolder

pic1 = os.path.join(app.config['UPLOAD_FOLDER'],'HERO.jpg')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
st=""
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/choice')
def choice():
    return render_template("choice.html")

@app.route('/upload',methods=["GET","POST"])
def upload():
    if request.method == "POST":
        file=request.files['uploadBills']
        #file.save(secure_filename(file.filename))
        #file.save(os.path.join("static/pics", file.filename))
        #some custom file name that you want
        if file and allowed_file(file.filename):
            st=allowed_file(file.filename)
            file.save("static/content/"+file.filename)
            time.sleep(3)
            p = "static/content/"+file.filename
            if  st == "png" or st == "jpg" or st == "jpeg":
                image=cv2.imread(p)
                lane_image = np.copy(image)
                canny_image= canny(lane_image)
                cropped_image = region_of_interest(canny_image)
                lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
                averaged_lines = average_slope_intercept(lane_image, lines)
                line_image = display_lines(lane_image,averaged_lines)
                combo_image= cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
                cv2.imshow("result",combo_image)
                cv2.waitKey(0)
            if st == "mp4":
                cap = cv2.VideoCapture(p)
                while(cap.isOpened()):
                    
                    _ , frame = cap.read()
                    
                    canny_image = canny(frame)
                    
                    cropped_image = region_of_interest(canny_image)
                    lines = cv2.HoughLinesP(cropped_image , 2 , np.pi/180 , 100 , np.array([]), minLineLength = 40 , maxLineGap = 5)
                    
                    averaged_lines = average_slope_intercept(frame , lines)
                    line_image = display_lines(frame , averaged_lines)
                    
                    combo_image = cv2.addWeighted(frame , 0.8, line_image , 1, 1)
                    
                    cv2.imshow('result', combo_image)
                    
                    if cv2.waitKey(40) & 0xFF== ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
        return render_template("result.html")
    return render_template("upload.html")

@app.route('/uploadobj',methods=["GET","POST"])
def uploadobj():
    if request.method == "POST":
        file=request.files['uploadBills']
        if file and allowed_file(file.filename):
            st=allowed_file(file.filename)
            file.save("static/content/"+file.filename)
            time.sleep(3)
            path = "static/content/"+file.filename
            if  st == "png" or st == "jpg" or st == "jpeg":
                yolo = YOLO()
                img =cv2.imread(path)
                oi = yolo.inference(img)
                
                cv2.imwrite("result.jpg",oi)
            if st == "mp4":
                cap = cv2.VideoCapture(path)
                print("OBJ DETECTION")
                outputFile ="Biking.avi"
                vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round       (cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                while cv2.waitKey(1) < 0:
                    ret,frame=cap.read()
                    if not ret:
                        print("Done processing !")
                        print("Output file is stored as ", outputFile)
                        cv2.waitKey(3000)
                        cap.release()
                        break
                    yolo = YOLO()
                    blob = cv2.dnn.blobFromImage(frame, 1/255, (yolo.inpWidth, yolo.inpHeight), [0,0,0], 1, crop=False)
                    yolo.net.setInput(blob)
                    outs=yolo.net.forward(yolo.getOutputsNames())
                    yolo.postprocess(frame,outs)
                    t, _ = yolo.net.getPerfProfile()
                    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                    vid_writer.write(frame.astype(np.uint8))
        return render_template("result.html")
    return render_template("uploadobj.html")

@app.route('/result')
def result():
    return render_template("result.html")

def allowed_file(filename):
    return filename.rsplit('.', 1)[1].lower() 

def make_coordinates(image, line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])
    
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit,axis= 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
    
    
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur= cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2=line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0 ,0), 10)      
    return line_image
    
def region_of_interest(image):
    height= image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

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
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=5)
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

if __name__ == "__main__":
   
    app.run()

