#importing all the important packages 
import numpy as np
import cv2 as cv 
import math 

REJECT_DEGREE_TH = 4
mouse_coordinates = (0, 0)
K = np.array([[910, 0, 582],
                     [0, 682.794, 437],
                     [0, 0, 1.0]])

vid = 4
file_loc = 'labeled/' + str(vid) + '.hevc'

hor = []
vertical = []

def get_py_from_vp(u_i, v_i, K):
    p_infinity = np.array([u_i, v_i, 1])
    K_inv = np.linalg.inv(K)
    r3 = K_inv @ p_infinity    
    r3 /= np.linalg.norm(r3)
    yaw = -np.arctan2(r3[0], r3[2])
    pitch = np.arcsin(r3[1])    
    
    return pitch, yaw

def mouse_callback(event, x, y, flags, param):
    global mouse_coordinates
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_coordinates = (x, y)
	
def FilterLines(Lines):
    FinalLines = []

    for Line in Lines:
        [[x1, y1, x2, y2]] = Line

        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        c = y2 - m*x2

        theta = math.degrees(math.atan(m))

        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            l = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)   
            FinalLines.append([x1, y1, x2, y2, m, c, l])

    if len(FinalLines) > 20:
        FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)
        FinalLines = FinalLines[:20]

    return FinalLines

def GetVanishingPoint(Lines):
    VanishingPoint = None
    MinError = 100000000000

    for i in range(len(Lines)):
        for j in range(i+1, len(Lines)):
            m1, c1 = Lines[i][4], Lines[i][5]
            m2, c2 = Lines[j][4], Lines[j][5]

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(Lines)):
                    m, c = Lines[k][4], Lines[k][5]
                    m_ = (-1 / m)
                    c_ = y0 - m_ * x0

                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_

                    l = math.sqrt((y_ - y0)**2 + (x_ - x0)**2)

                    err += l**2

                err = math.sqrt(err)

                if MinError > err:
                    MinError = err
                    VanishingPoint = [x0, y0]

    return VanishingPoint

def main():
    cap = cv.VideoCapture(file_loc)
    mean_vanishing_point_x = 0
    mean_vanishing_point_y = 0
    counter = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        image = frame
        image_height, image_width = image.shape[:2]
        
        roi_vertices = np.array([[(0, image_height),
                         (image_width, image_height),
                         (image_width * 1, image_height * 0.4),
                         (0, image_height * 0.4)]], dtype=np.int32)
        
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        

        mask = np.zeros_like(grayscale)
        cv.fillPoly(mask, roi_vertices, 255)
        masked_grayscale = cv.bitwise_and(grayscale, mask)
        cv.imshow('Frame2', masked_grayscale)

        blur = cv.GaussianBlur(masked_grayscale, (5, 5), 0)
        edges = cv.Canny(blur, 20, 50)
        cv.imshow('Frame3', edges)
        # lines = cv.HoughLines(edges, 1, np.pi/180, 200)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, 5, 15)
        print(lines)
        finallines = FilterLines(lines)
        vanishing_point = GetVanishingPoint(finallines)
        
        for Line in finallines: 
            cv.line(image, (Line[0], Line[1]), (Line[2], Line[3]), (0, 255, 0), 2)
        
        if vanishing_point is None:
            print("Vanishing Point not found.")
            counter+=1
        else: 
            # mean_vanishing_point_x+= vanishing_point[0]
            # mean_vanishing_point_y+= vanishing_point[1]
            hor.append(vanishing_point[0])
            vertical.append(vanishing_point[1])
            counter+=1 
        cv.setMouseCallback("Frame", mouse_callback)
        u = np.median(hor)
        v = np.median(vertical)
        # if vanishing_point is None: 
        #     u = 582
        #     v = 437
        # else: 
        #     if vanishing_point[0]> 582+50 or vanishing_point[0]<582-50: u = 582
        #     else: u = vanishing_point[0]
        #     if vanishing_point[1]> 437+50 or vanishing_point[1]<437-50: v = 437
        #     else: v = vanishing_point[1]
        print(counter,"Mouse coordinates: (x={}, y={})".format(mouse_coordinates[0], mouse_coordinates[1]))
            
        cv.circle(image, (mouse_coordinates[0], mouse_coordinates[1]), 10, (0, 255, 0), -1)
        cv.circle(image, (int(u), int(v)), 10, (0, 0, 255), -1)
        
        cv.imshow('Frame', image)
        if mouse_coordinates[0] == 0 and mouse_coordinates[1] == 0: 
            pitch, yaw = get_py_from_vp(u,v,K)
        else: 
            pitch, yaw = get_py_from_vp(mouse_coordinates[0],mouse_coordinates[1],K)
        formatted_pitch = "{:.7e}".format(-pitch)
        formatted_yaw = "{:.7e}".format(-yaw)
        loc = 'data/' + str(vid) + '.txt'
        val = formatted_pitch + " " + formatted_yaw
        with open(loc, "a") as file:
            file.write(val + '\n')
        
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == "__main__": 
	main()