REQUIREMENTS
1. Python
2. OpenCV (pip install --user opencv-contrib-python)
3. pandas (pip install pandas) 
4. numpy  (pip install numpy)
5. pillow (pip install pillow)


HOW TO USE
1)Register a new student by vlivking on the register button.
2)The webcam will open up and the program will take multiple images of the face of the student. These will be stored in the 'data' directory.
3)'details.csv' file will be created with the name and ID of the student.
4)Click on the train button to train the classifier with the faces of the student, taken from the data directory.
5)Now click on take attendance whenever you want to mark the attendance and the webcam will open up, and the students who show their faces will be marked as present and the file will be stored in the attendance folder.