Comp 4301 Final Project: Object detection and Navigation

Made by: Mohammed Shoaib, Yaksh Haranwala, Salman Haidri, Abdul Shaji

Run Instructions:

After you have downloaded the code,
Run=> pip install -r requirements.txt
This will install all needed modules to run the project
Note: we are using pytorch with GPU in order to run this, although not necessary it runs a lot faster with GPU.

As long as the device running the code has a webcam, you can simply run the program using the command: Python Runner.py
If a phone camera is not detected, it will automatically switch to the device local webcam.

To use with phone camera, download 'IP Webcam' from the playstore in an android device.
1. Scroll down in the app and hit Start server
2. Make sure the the device running the code and the phone are on the same network.
3. Modify the URL in the code to match the URL from IP webcam app after starting server.
4. Run the program using: Python Runner.py

The video config we have used is landscape, 720x480 resolution, with frames capped at 30fps, using a wide angle camera.
Once the program runs, it will display two windows, one for MiDas depth view, and one for live feed with object detection.

To Exit, in the window displaying the live video feed, press Q