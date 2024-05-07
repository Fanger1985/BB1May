
>>> %Run camtest.py
Camera 0 is working!
[ WARN:0@1.263] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video1): can't open camera by index
[ERROR:0@1.267] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
Error: Camera 1 could not be opened.
[ WARN:0@1.267] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video2): can't open camera by index
[ERROR:0@1.271] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
Error: Camera 2 could not be opened.
[ WARN:0@1.272] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video3): can't open camera by index
[ERROR:0@1.275] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
Error: Camera 3 could not be opened.[ WARN:0@1.276] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video4): can't open camera by index
[ERROR:0@1.279] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.280] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video5): can't open camera by index
[ERROR:0@1.283] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.284] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video6): can't open camera by index
[ERROR:0@1.288] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.289] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video7): can't open camera by index
[ERROR:0@1.292] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.293] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video8): can't open camera by index
[ERROR:0@1.297] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.298] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video9): can't open camera by index
[ERROR:0@1.302] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.303] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video10): can't open camera by index
[ERROR:0@1.306] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.308] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video11): can't open camera by index
[ERROR:0@1.311] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.312] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video12): can't open camera by index
[ERROR:0@1.315] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range
[ WARN:0@1.316] global cap_v4l.cpp:997 open VIDEOIO(V4L2:/dev/video13): can't open camera by index
[ERROR:0@1.320] global obsensor_uvc_stream_channel.cpp:159 getStreamChannelGroup Camera index out of range

Error: Camera 4 could not be opened.
Error: Camera 5 could not be opened.
Error: Camera 6 could not be opened.
Error: Camera 7 could not be opened.
Error: Camera 8 could not be opened.
Error: Camera 9 could not be opened.
Error: Camera 10 could not be opened.
Error: Camera 11 could not be opened.
Error: Camera 12 could not be opened.
Error: Camera 13 could not be opened.
[ WARN:0@11.337] global cap_v4l.cpp:1134 tryIoctl VIDEOIO(V4L2:/dev/video14): select() timeout.
Camera 14 failed to capture a frame.



import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"Camera {index} is working!")
        else:
            print(f"Camera {index} failed to capture a frame.")
    else:
        print(f"Error: Camera {index} could not be opened.")

# Test a range of indices
for i in range(32):  # Adjust this range if necessary
    test_camera(i)
