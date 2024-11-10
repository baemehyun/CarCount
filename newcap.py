# import modules 
import cv2 
# import files 

# open file 
cap = cv2.VideoCapture('/Users/mean/year-4/carCount/Car-Detection-and-Car-Counter/SB-BR-023-05_20240403101351548.mp4') 

# get FPS of input video 
fps = cap.get(cv2.CAP_PROP_FPS) 

# define output video and it's FPS 
output_file = 'output1-full.mp4'
output_fps = fps 

# define VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)  # Half of original width
new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)  # Half of original height
new_size = (960, 540)
# out = cv2.VideoWriter(output_file, fourcc, output_fps, new_size) 
out = cv2.VideoWriter(output_file, fourcc, output_fps, 
					(int(cap.get(3)), int(cap.get(4)))) 
# read and write frams for output video 
# while cap.isOpened(): 
# 	ret, frame = cap.read() 
# 	if not ret: 
# 		break

# 	out.write(frame) 

frame_skip = 10  # Increase this value to reduce more frames

# Read and write frames for output video with resizing and frame skipping
frame_count = 0  # Initialize frame count
while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break

    # Skip frames according to frame_skip value
    if frame_count % frame_skip == 0:  # Only keep frames that match this condition
        # Resize the frame
        # resized_frame = cv2.resize(frame, new_size)  
        # Write the resized frame to the output video
        out.write(frame) 
        print(f"complete add frame {frame_count}")
    frame_count += 1 

# release resources 
cap.release() 
out.release() 
cv2.destroyAllWindows() 

# download output video on local machine 
# files.download(output_file) 
