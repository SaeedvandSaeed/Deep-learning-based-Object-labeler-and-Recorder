import cv2
from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from pygame import mixer

from PIL import Image
from sort import *

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device) #auto select
model.eval()

classes = utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
  
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = '../data/video/overpass.mp4'

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

#vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

#cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#ret,frame=vid.read()
cap = cv2.VideoCapture(1)
ret, frame = cap.read()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#vh = frame.shape[0]
print ("Video size", frame_width,frame_height)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(frame_width,frame_height))

# ------------------Folder management and save--------------


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
record = False
frame_counter = 0

initial_path = os.getcwd() 
path = initial_path + "/Ceptured"
folder_count = len(os.walk(path).__next__()[1]) + 1
total_file_count = len(os.walk(path).__next__()[2]) + 1

if(folder_count > 1):
    if(len(os.walk(path + "/cap" + str(folder_count- 1)).__next__()[2]) == 0):
        folder_count = folder_count - 1

folderName = "cap" + str(folder_count)

if(not os.path.exists(path+"/"+folderName)):
    os.mkdir(path+"/"+folderName)

# out = cv2.VideoWriter('Videos/cap-' + str(folder_count)+'.avi',
#                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

mixer.init()
sound = True
start_time = time.time()
#-----------------------------------------------------------

frames = 0
keyboard_class=0
mouse_class=0
laptop_class=0
selection_star_pos = 0
selected_class_change = 0

starttime = time.time()
while(True):
    ret, frame = cap.read()

    if not ret:
        break
    #-------------------------------------
  
    if(record == True):
        if(sound):
            mixer.music.load("Sounds/button37.mp3")
            mixer.music.play()
            sound = False
            start_time = time.time()
            video_count = len(os.walk(os.getcwd() + "/Videos").__next__()[2])
            print(str(os.walk(initial_path + "/Videos")))
            print(folder_count)
            out = cv2.VideoWriter('Videos/cap-' + str(folder_count)+"-"+str(video_count)+'.avi', 
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

        out.write(frame)

        if(frame_counter % 5 == 0):
            cv2.imwrite(path+"/" + folderName + "/cap-"+ str(folder_count) + "-" + str(frame_counter) + ".jpg", frame)

    else:
        if(not sound):
            mixer.music.load("Sounds/button-48.mp3")
            mixer.music.play()
            sound = True
        sound = True

        cv2.putText(frame, '.....', (520, 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (0, 0, 0),
                    lineType=cv2.LINE_AA)

    #--------------------------Object detection and recorder------------------------------

    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        craped_frames = []

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
         
            # Crop and save each object
            crop = frame[ y1:y1 + box_h, x1:x1 + box_w]
            #print (cls)

            if(record == True and frame_counter % 5 == 0):
                if(not os.path.exists(path +"Samples/" + str(cls))):
                    os.mkdir(path +"Samples/" + str(cls))
                if(cls == 'keyboard'):
                    if(not os.path.exists(path +"Samples/" + str(cls))):
                        os.mkdir(path +"Samples/" + str(cls))
                    if(not os.path.exists(path +"Samples/" + str(cls)+ "/" + str(keyboard_class))):
                        os.mkdir(path +"Samples/" + str(cls)+ "/" + str(keyboard_class))
                    file_count = len(os.walk(path +"Samples/" + str(cls)+"/"+ str(keyboard_class)).__next__()[2])
                    cv2.imwrite(path + "Samples/" + str(cls) +"/"+ str(keyboard_class)+"/cap-"+ 
                        str(folder_count) + "-" + str(frame_counter) + "-" + str(file_count) + 
                        "(" + str(x1) +"," + str(y1) +"," + str(box_w) +"," + str(box_h) + ")" +
                        ".jpg", crop)
                elif(cls == 'laptop'):
                    if(not os.path.exists(path +"Samples/" + str(cls))):
                        os.mkdir(path +"Samples/" + str(cls))
                    if(not os.path.exists(path +"Samples/" + str(cls)+ "/" + str(laptop_class))):
                        os.mkdir(path +"Samples/" + str(cls)+ "/" + str(laptop_class))
                    file_count = len(os.walk(path +"Samples/" + str(cls)+"/"+ str(laptop_class)).__next__()[2])
                    cv2.imwrite(path + "Samples/" + str(cls) +"/"+ str(laptop_class)+"/cap-"+ 
                        str(folder_count) + "-" + str(frame_counter) + "-" + str(file_count) + 
                        "(" + str(x1) +"," + str(y1) +"," + str(box_w) +"," + str(box_h) + ")" +
                        ".jpg", crop)
                elif(cls == 'mouse'):
                    if(not os.path.exists(path +"Samples/" + str(cls))):
                        os.mkdir(path +"Samples/" + str(cls))
                    if(not os.path.exists(path +"Samples/" + str(cls)+ "/" + str(mouse_class))):
                        os.mkdir(path +"Samples/" + str(cls) + "/" + str(mouse_class))
                    file_count = len(os.walk(path +"Samples/" + str(cls)+"/"+ str(mouse_class)).__next__()[2])
                    cv2.imwrite(path + "Samples/" + str(cls) +"/"+ str(mouse_class)+"/cap-"+ 
                        str(folder_count) + "-" + str(frame_counter) + "-" + str(file_count) + 
                        "(" + str(x1) +"," + str(y1) +"," + str(box_w) +"," + str(box_h) + ")" +
                        ".jpg", crop)
                else:
                    file_count = len(os.walk(path +"Samples/" + str(cls)+"/").__next__()[2])
                    cv2.imwrite(path + "Samples/" + str(cls) +"/cap-"+ 
                        str(folder_count) + "-" + str(frame_counter) + "-" + str(file_count) + 
                        "(" + str(x1) +"," + str(y1) +"," + str(box_w) +"," + str(box_h) + ")" +
                        ".jpg", crop)
            
            if(record == True):
                elapsed_time = time.time() - start_time
                cv2.putText(frame, 'Rec:' + str(int(elapsed_time)), (520, 40),#460
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 0, 0),
                        lineType=cv2.LINE_AA)
                
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
               
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    #-----------------------------------------------------------------------
    key = cv2.waitKey(1)

    cv2.putText(frame, 'Scenario: ', (20, 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                1, (0, 250, 0),
                lineType=cv2.LINE_AA)
    cv2.putText(frame, folderName, (180, 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                1, (0, 0, 250),
                lineType=cv2.LINE_AA)

    cv2.putText(frame, "key_c: " + str(keyboard_class), (frame_width - 200, 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                1, (0, 0, 0),
                lineType=cv2.LINE_AA)
    cv2.putText(frame, "lap_c: " + str(laptop_class), (frame_width - 200, 65),
                cv2.FONT_HERSHEY_TRIPLEX,
                1, (0, 0, 0),
                lineType=cv2.LINE_AA)
    cv2.putText(frame, "mou_c: " + str(mouse_class), (frame_width - 200, 90),
                cv2.FONT_HERSHEY_TRIPLEX,
                1, (0, 0, 0),
                lineType=cv2.LINE_AA)
        
    if(selected_class_change == 0):
        selection_star_pos = 45
    elif (selected_class_change == 1):
        selection_star_pos = 70
    elif (selected_class_change == 2):
        selection_star_pos = 95

    cv2.putText(frame, "*", (frame_width - 220, selection_star_pos),
                cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 0, 0),
                lineType=cv2.LINE_AA)

    frame = cv2.resize(frame, (frame_width//2, frame_height//2),
                interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)

    if key == 113:  # q
        break
    elif key == 114:  # r
        record = True
        print("Recording")
    elif key == 115:  # s
        record = False
        print("Stoped.")
    elif key == 112:  # p
        if(record == False and folder_count > 1):
            folder_count = folder_count-1
            folderName = "cap" + str(folder_count)
            print("Prev.")
    elif key == 110:  # n
        if(record == False and folder_count < len(os.walk(path).__next__()[1])):
            folder_count = folder_count+1
            folderName = "cap" + str(folder_count)
            print("Next.")
    elif key == 97:  # a
        if(record == False):
            if(len(os.walk(path+"/" + folderName).__next__()[2]) == 0):
                print(
                    'Before adding new directory, record to current one please!')
                continue
            elif (folder_count < len(os.walk(path).__next__()[1])-1):
                print('go to the final directory with n key.')
                continue
            else:
                folder_count = folder_count+1
                folderName = "cap" + str(folder_count)
                if(not os.path.exists(path + "/" + folderName)):
                    os.mkdir(path+"/"+folderName)
                print("New.")
    elif key == 46:  # .
        print(selected_class_change)
        selected_class_change=selected_class_change+1
        if(selected_class_change > 2):
            selected_class_change = 0
    elif key == 43: #"+"
        if(selected_class_change == 0):
            keyboard_class = keyboard_class + 1
        elif (selected_class_change == 1):
            laptop_class = laptop_class + 1
        elif (selected_class_change == 2):
            mouse_class = mouse_class + 1
    elif key == 45: #"-"
        if(selected_class_change == 0):
            if(keyboard_class != 0):
                keyboard_class = keyboard_class - 1
        elif (selected_class_change == 1):
            if(laptop_class != 0):
                laptop_class = laptop_class - 1
        elif (selected_class_change == 2):
            if(mouse_class != 0):
                mouse_class = mouse_class - 1

    frame_counter = frame_counter + 1

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
