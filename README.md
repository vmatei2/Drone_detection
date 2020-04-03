# Drone detection
Drone-detection program that differentiates between drone signals and signals coming from WI-FI or Bluetooth.


Although they operate in the same frequency spectrum, they can be distinguished by after storing the signals and perfroming singal analysis techniques.
This is a program that takes as input the CSV file, which has stored several features for different signal types(Wifi, BT and Drone). Then, the data is used to train and evaluate machine learning algorithms to be used in drone detection. 
All the data in the CSV file was captured using a HackRF-One. The drone used was a Parot Bepop Dron 2 and the signals were captured inside an anechoic chamber. 

Worked on this research project as part of the SURE Scheme at the University of Sheffield.
