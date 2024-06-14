import firebase_admin
from firebase_admin import credentials,db,initialize_app,storage
import io
from PIL import Image
import face_recognition
import datetime
import numpy as np
import cv2
import requests
from tensorflow import keras
from keras import Sequential,models
import pathlib
from fastapi import FastAPI,UploadFile
from statistics import mode
import google.generativeai as genai
from utils_jarkom.datasets import get_labels
from utils_jarkom.inference import detect_faces
from utils_jarkom.inference import draw_text
from utils_jarkom.inference import draw_bounding_box
from utils_jarkom.inference import apply_offsets
from utils_jarkom.inference import load_detection_model
from utils_jarkom.preprocessor import preprocess_input
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import uvicorn
import json 
import mysql.connector
from datetime import datetime, timezone


app=FastAPI()
counter=0

mydb = mysql.connector.connect(
  host="localhost",
  user="<UR USERNAME>",
  password="PASSWORD",
  database="UR DATABASE"
)

def my_jsonify(cursor,result):
   row_headers=[x[0] for x in cursor.description]
   json_data=[]
   for result in result:
        json_data.append(dict(zip(row_headers,result)))
   return jsonable_encoder(json_data)

genai.configure(api_key="<APIKEY>")
model = genai.GenerativeModel('gemini-1.5-flash')

emotion_model_path = r'D:\learning\firebase\fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
detection_model_path = 'haarcascade_frontalface_default.xml'
# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = models.load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)


cred = credentials.Certificate(r'<CREDENTIALS>')
firebase_admin.initialize_app(cred,{
    'storageBucket': '<UR BUCKET'
})

hasan_image = face_recognition.load_image_file("person_image/my_face.jpg")
hasan_face_encoding = face_recognition.face_encodings(hasan_image)[0]
omar_image = face_recognition.load_image_file("person_image/omar_face.jpg")
omar_face_encoding = face_recognition.face_encodings(omar_image)[0]

known_face_encodings = [
    hasan_face_encoding,
    omar_face_encoding
]
known_face_names = [
    "hasan",
    "skibidi rizz"
]

def upload_image(image_path, destination_blob_name):
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(image_path)
    blob.make_public()
    print(f'File {image_path} uploaded to {destination_blob_name}.')
    print(f'Public URL: {blob.public_url}')
    return blob.public_url

def decode_image(byte_string):
    image_stream = io.BytesIO(byte_string)
    image = Image.open(image_stream)
    timee=f"{datetime.datetime.now()}"
    image_name= timee+".jpeg"
    image_name = image_name.replace(':', '_')
    save_path = fr"D:\learning\firebase\temp_image\{image_name}"
    image.save(save_path, format='JPEG')
    return save_path,timee

def compare(known_face_encodings,image_path):
    face_names = []
    image=cv2.imread(image_path, cv2.IMREAD_COLOR)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            ############ FACE EVAL###################
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else :
                 name="unknown"
            face_names.append(name)

            ########### EVALUATION DONE#################

    emotion_list=[]
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        ######### EMOTION EVAL##############
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x1=left-40
        x2=right+30
        y1=top-72
        y2=bottom+40
        gray_face = gray_image[y1:y2, x1:x2]
        print(x1,x2,y1,y2)
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        print(emotion_text)

        ######### END EMOTION EVAL ########
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 30), (right, bottom+22), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"name : {name}", (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        cv2.putText(image, f"emotion : {emotion_text} prob: {emotion_probability:.2f}", (left + 6, bottom + 20), font, 0.4, (255, 255, 255), 1)
        emotion_list.append({"name":f"{name}","emotion":f"{emotion_text}","emotion_probability":f"{emotion_probability}"})
    cv2.imwrite(image_path,image)
    print (face_names)
    return face_names,emotion_list

def eval_emotion(image_path):
    image=cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations_lib1 = face_recognition.face_locations(image)
    faces = detect_faces(face_detection, gray_image)

    print(face_locations_lib1)
    print(faces)

    for (top, right, bottom, left)in face_locations_lib1:
        x1=left-40
        x2=right+30
        y1=top-72
        y2=bottom+40
        gray_face = gray_image[y1:y2, x1:x2]
        
        print(x1,x2,y1,y2)
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        print(emotion_text)
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"{emotion_text} : {round(emotion_probability,2)}", (x1+ 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imwrite(image_path,image)
    return image_path

def AI_description(image_path,prompt="give me a creative description of this image, tell me about his name and his current emotion too. dont use ' in the sentence"):
    cookie_picture = {
    'mime_type': 'image/png',
    'data': pathlib.Path(image_path).read_bytes()
    }
    print(pathlib.Path(image_path).read_bytes())
    response = model.generate_content(
        contents=[prompt, cookie_picture]
        )
    return response.text

@app.post("/upload_database/")
async def upload_image_to_database():
    x = requests.get('http://192.168.1.24:81/jpeg')
    image_bytes = x.content
    path,time= decode_image(image_bytes)
    face_list,emot_list= compare(known_face_encodings,path)
    AI_desc=AI_description(path)
    url= upload_image(path,fr"ESP32CAM/{time}.jpeg")
    mycursor = mydb.cursor()
    stringfied_emot_list=[]
    for member in emot_list:
        temp_str = json.dumps(member)
        stringfied_emot_list.append(temp_str)
    mycursor.execute(f"INSERT INTO esp32_handler (ID, TIME, EMOTION, DETECTION, IMAGE_URL, AI_DESCRIPTION) VALUES ('{counter}', '{time}','{face_list}','{stringfied_emot_list}', '{url}', '{AI_desc}')")
    counter=counter+1

@app.post("/upload_database/upload")
async def create_upload_file(file: UploadFile):
    if file.size> 5000000:
        return {"message":"size limit reached"}
    elif file.size> 1000000:
        try:
            with open("temp_file/"+file.filename, 'wb') as f:
                while contents := file.file.read(1024 * 1024):
                    f.write(contents)
        except Exception as e:
            print(e)
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
    else :
        try:
            with open(fr'D:\learning\firebase\temp_file\{file.filename}', 'wb') as f:
                while contents := file.file.read():
                    f.write(contents)
        except Exception as e:
            print(e)
            return {"message": "There was an error uploading the file"}
        
    path=fr'D:\learning\firebase\temp_file\{file.filename}'
    timestamp= datetime.now(timezone.utc)
    time=timestamp.strftime('"%Y-%m-%d %H_%M_%S"')
    face_list,emot_list= compare(known_face_encodings,path)
    AI_desc=AI_description(path)
    AI_desc_removed = AI_desc.replace("'","")
    print(AI_desc_removed)
    url= upload_image(path,fr"ESP32CAM/{time}.jpeg")
    mycursor = mydb.cursor()
    stringfied_emot_list=[]
    list_srting=""
    face_string=""
    for member in emot_list:
        temp_str = json.dumps(member)
        stringfied_emot_list.append(temp_str)
    for member in stringfied_emot_list:
        list_srting= list_srting+f"{member}"
    for member in face_list:
        face_string= face_string+f"{member}"


    query = "INSERT INTO esp32_handler (TIME, EMOTION, DETECTION, IMAGE_URL, AI_DESCRIPTION) VALUES (%s, %s, %s, %s, %s)"
    values = (time, list_srting, face_string, url, AI_desc_removed)
    mydb.commit()
    mycursor.execute(query, values)
    return {"message": f"Successfuly added","url": f"{url}"}

def my_jsonify(cursor,result):
   row_headers=[x[0] for x in cursor.description]
   json_data=[]
   for result in result:
        json_data.append(dict(zip(row_headers,result)))
   return jsonable_encoder(json_data)

@app.get("/upload_database/retrieve")
async def retrieve_table ():
    mycursor = mydb.cursor()
    mycursor.execute(f"SELECT * FROM  esp32_handler")
    myresult = mycursor.fetchall()
    return my_jsonify(mycursor,myresult)


if __name__ == "__main__":
    #packetriot tunnel is connected to port 80 -- config it in ur own terminal
    uvicorn.run(app=app,host="0.0.0.0",port=90)
                                                                                                                 