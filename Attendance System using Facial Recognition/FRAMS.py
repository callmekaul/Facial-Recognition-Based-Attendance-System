import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image
from datetime import datetime
import time
import os
from os import path
import csv
import cv2


mainWin = Tk()
mainWin.title("Attendance System")
mainWin.configure(background = 'black')
Label(mainWin, text = 'Facial Recognition Based Attendance Management System', fg = 'white', bg = 'black',font = ('arial',32,'bold')).pack()

def error():
    messagebox.showinfo("Error", "Please enter valid details")


def regSubButtonClick():                                                                        # function called when submit button clicked in registration window
    sName=sNameEnt.get()
    sID=sIDEnt.get()
    if sName == '' or sID == '':
        error()
    else:
        cam = cv2.VideoCapture(0)
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        imgID = 0
        
        while True:
            _,img = cam.read()                                                                  # _ not used
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(gray, 1.3, 5)                                   # returns x, y, w, h around face
            for(x, y, w, h) in faces:
                cv2.rectangle(img,(x,y),(x + w,y + h),(255, 0, 0), 2)
                imgID += 1                                                                      # increment image ID
                cv2.imwrite("data/id."+str(sID)+"."+str(imgID)+".jpg",gray[y:y + h, x:x + w])   # writing image to dataset
                cv2.imshow('Frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):                                               # wait for 100 miliseconds, quit if q pressed
                break
            elif imgID > 19:                                                                    # exit if more than 70 images
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [sID,sName]
        with open('details.csv','a+',newline = '') as csvFile:                                  # writing the student details to 'details.csv'
            writer = csv.writer(csvFile)
            writer.writerow(row)
            csvFile.close()


def regButtonClick():                                                                           # function called when register button is clicked
    regWindow = Tk()
    regWindow.title("New Student Registration")
    regWindow.configure(background = 'black')
    regWindow.geometry('720x540')

    if not path.exists('data'):
        os.mkdir('data')

    if not path.exists('details.csv'):
        with open ('details.csv','w', newline = '') as temp:
            tWriter = csv.writer(temp)
    
    Label(regWindow, text = 'Register a new Student', fg = 'white', bg = 'black',font = ('arial',32,'bold')).pack()
    
    Label(regWindow, text = 'Enter student name', fg = 'white', bg = 'black',font = ('arial',12,'bold')).pack()
    
    global sNameEnt
    sNameEnt = tk.Entry(regWindow, fg = 'white', bg = 'black',font = ('arial',12,'bold'))
    sNameEnt.pack()
    
    Label(regWindow, text = 'Enter student ID', fg = 'white', bg = 'black',font = ('arial',12,'bold')).pack()
    
    global sIDEnt
    sIDEnt = tk.Entry(regWindow, fg = 'white', bg = 'black',font = ('arial',12,'bold'))
    sIDEnt.pack()

    regSubButton = tk.Button(regWindow, text = 'Submit', command = regSubButtonClick, fg = 'White', bg = 'black', font = ('arial',16,'bold')).pack()


def trainClassifier():                                                                          # function called when train button is clicked
    path = [os.path.join('data',f) for f in os.listdir('data')]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')                                                    # convert all images in the data directory to numpy arrays
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[-1].split(".")[1])                                        # retrieve id of the image using split() funciton on the file names
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()                                                  # create custom classifier and train it with the custom faces and ids
    clf.train(faces, ids)
    clf.write("classifier.xml")


def takeAttendance():                                                                                       # function called when submit button pressed in subject window
    sName = subEnt.get()
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = 'attendance/' + date + '/' + sName + "_" + Hour + "-" + Minute + "-" + Second + ".csv"       # path where the record will be stored

    if not path.exists('attendance'):
        os.mkdir('attendance')

    if not path.exists('attendance/' + date):
        os.mkdir('attendance/' + date)
        
    with open (fileName,'w', newline = '') as temp:
        tWriter = csv.writer(temp)
        tWriter.writerow(['ID','Student','Attendance','Time'])
    
    with open('details.csv') as csvFile:                                                                    # copying student details from details.csv and marking each student as absent by default
        with open (fileName,'a') as temp:
            reader = csv.DictReader(csvFile, fieldnames = ['ID','Student'])
            fieldnames = ['ID','Student','Attendance','Time']
            writer = csv.DictWriter(temp, fieldnames = fieldnames)
            for row in reader:
                id1 = str(row['ID'])
                stud1 = str(row['Student'])
                writer.writerow({'ID' : id1, 'Student' : stud1, 'Attendance' : 'Absent','Time' : 'Absent'})


    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")                                                                              # loading custom classifier

    cam = cv2.VideoCapture(0)

    while True:
        _,img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, 1.3, 5)
        coords = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,260,0), 2)
            id, _ = clf.predict(gray[y:y+h, x:x+w])
            cv2.putText(img, str(id), (x+h, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 260, 0), 4)                # drawing rectangle around face and labeliing it with the respective face id
            cv2.imshow('Frame', img)
            with open (fileName,'r+') as temp:
                reader = csv.DictReader(temp, fieldnames = ['ID','Student','Attendance','Time'])
                writer = csv.DictWriter(temp, fieldnames = ['ID','Student','Attendance','Time'])
                for row in reader:                                                                          # comparing the id of the face in the frame with every id in the .csv file
                    id1 = str(row['ID'])                                                                    # and inserting a row with the attendance marked as present each time a match is found
                    stud1 = str(row['Student'])
                    if str(id) == str(row['ID']):
                        if str(row['Attendance']) == 'Present':
                            continue
                        writer.writerow({'ID' : id1, 'Student' : stud1, 'Attendance' : 'Present','Time' : str(datetime.fromtimestamp(ts).strftime('%H:%M:%S'))})
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    df = pd.read_csv(fileName)                                                                              # loading csv file to a pandas dataframe
    df.drop_duplicates(['ID'], keep = 'last', inplace = True)                                               # removing all duplicate rows based on id (multiple rows for present students)
    df = df.sort_values('ID')
    df.to_csv(fileName, index = False)
    
    
    cam.release()
    cv2.destroyAllWindows()


def attButtonClick():                                                                           # function called when take attendance button is clicked
    subWin = Tk()
    subWin.title("Enter the Subject")
    subWin.configure(background = 'black')
    Label(subWin, text = 'Subject:- ', fg = 'white', bg = 'black',font = ('arial',16,'bold')).pack()
    global subEnt
    subEnt = tk.Entry(subWin, fg = 'white', bg = 'black', font = ('arial',16,'bold'))
    subEnt.pack()
    subSubButton = tk.Button(subWin, text = 'Submit', command = takeAttendance, fg = 'White', bg = 'black', font = ('arial',16,'bold')).pack()


regButton = tk.Button(mainWin, text = 'Register', command = regButtonClick, fg = 'White', bg = 'black', font = ('arial',16,'bold'), height = 8, width = 40).pack()
attButton = tk.Button(mainWin, text = 'Take Attendance', command = attButtonClick, fg = 'White', bg = 'black',font = ('arial',16,'bold'), height = 8, width = 40).pack()
trainButton = tk.Button(mainWin, text = 'Train', command = trainClassifier, fg = 'White', bg = 'black',font = ('arial',8,'bold'), height = 4, width = 20).pack(side = RIGHT)

mainWin.mainloop()