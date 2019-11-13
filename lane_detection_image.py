#---------------------------模板库声明---------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#----------------------------函数声明---------------------------------
def abs_sobel_thresh(img,thresh_min=50, thresh_max=100):   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & \
        (scaled_sobel <= thresh_max)]= 100
    binary_output[(scaled_sobel < thresh_min) | \
        (scaled_sobel > thresh_max)]= 0
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh_min=0,thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 100
    binary_output[(gradmag < thresh_min) | (gradmag > thresh_max)] = 0

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 100
    binary_output[(absgraddir < thresh[0]) | (absgraddir > thresh[1])] = 0

    # Return the binary image
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 100
    binary_output[(channel < thresh[0]) | (channel > thresh[1])] = 0
    return binary_output

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 100
    binary_output[(b_channel < thresh[0]) | (b_channel > thresh[1])] = 0
    return binary_output
def thresholding(img):
    #setting all sorts of thresholds
    x_thresh = abs_sobel_thresh(img, thresh_min=50 ,thresh_max=100)
    mag_thresh1 = mag_thresh(img, sobel_kernel=3, thresh_min=50, thresh_max=150)
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = hls_select(img, thresh=(180, 255))
    lab_thresh = lab_select(img, thresh=(80, 100))
    luv_thresh = luv_select(img, thresh=(225, 255))

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 100) & (mag_thresh1 == 100)) | \
        ((dir_thresh == 100) & (hls_thresh == 100)) | (lab_thresh == 100) | \
        (luv_thresh == 100)] = 255

    return threshholded

def aerial_view(img):
    src=np.float32([[(0, 360), (0, 720), (1280, 360), (1280, 720)]])
    dst = np.float32([[(0, 0), (0, 720), (1280, 0), (1280, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    binary_warped = cv2.warpPerspective(img, M, img.shape[1::-1], \
       flags=cv2.INTER_LINEAR)
    return binary_warped

def aerial_view_inverse(img):
    src=np.float32([[(0, 360), (0, 720), (1280, 360), (1280, 720)]])
    dst = np.float32([[(0, 0), (0, 720), (1280, 0), (1280, 720)]])
    M = cv2.getPerspectiveTransform(dst, src)
    binary_warped = cv2.warpPerspective(img, M, img.shape[1::-1], \
       flags=cv2.INTER_LINEAR)
    return M

def find_line(binary_warped):
    divide=0.5
    #去掉雷达引起的特征点
    binary_warped[550:700,430:830]=0
    #去除上一半图片
    tmp_img=binary_warped.copy()
    binary_warped[0:360,0:1280]=0
    half_nonzero = binary_warped.nonzero()
    half_nonzeroy = np.array(half_nonzero[0])#返回非零行位置array
    if len(half_nonzeroy)< 200 :
        binary_warped=tem_img.copy()
    if len(half_nonzeroy)> 200 :
        divide=0.75
    # Take a histogram of the bottom half of the image,按列相加
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]*divide):,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    binary_warped=tmp_img.copy()
    nonzero = binary_warped.nonzero() #矩阵中非零元素的位置
    nonzeroy = np.array(nonzero[0])  #创造数组
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    #rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 180
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    #right_lane_inds = []
    #声明矩形图像
    rectangle_img=np.zeros((720,1280,1),dtype="uint8")
    #声明左右增加量
    left_margin=margin
    right_margin=margin
    #
    decider=0
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        #win_xright_low = rightx_current - margin
        #win_xright_high = rightx_current + margin

        #画出矩形
        cv2.rectangle(rectangle_img,(leftx_current-left_margin,win_y_high), \
            (leftx_current+right_margin,win_y_low),(225,0,225),1)
        # Identify the nonzero pixels in x and y within the window
        #nonzerox与y进行并运算，得到新的数组
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
        (nonzerox >= leftx_current-left_margin) &  (nonzerox < leftx_current+right_margin)).nonzero()[0]
    
        #good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
        #(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        left_lane = np.concatenate(left_lane_inds)
        #right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if (len(good_left_inds) > minpix | decider>0):
            decider=1
            print("window")
            print(window)

            print("slope")
            
            #对点进行直线拟合确定斜率
            slope=np.polyfit(nonzerox[left_lane], nonzeroy[left_lane], 1)
            print(slope[0])
            left_margin=np.int(-slope[0]*100)
            print(left_margin)
            right_margin=np.int(-slope[0]*200)
            #end_point_x=pts_left[0][0][0] #矩阵行列，列表第一个元素
            #end_point_y=pts_left[0][0][1]
            #start_point_x=pts_left[0][-1][0]
            #start_point_y=pts_left[0][-1][1]
            #slope=(end_point_y-start_point_y)/(end_point_x-start_point_x)
            #print(pts_left)
            leftx_current = np.int(np.mean(nonzerox[good_left_inds])) #获取窗口中非零点x的索引的均值索引，作为base_point
        #if len(good_right_inds) > minpix:        
            #rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
       
    line_image=cv2.addWeighted(rectangle_img, 1, binary_warped, 1, 0)
    cv2.imshow("rectangle_img",line_image)
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    #right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    lefty_min=min(lefty)
    leftx_max=max(leftx)
    print("min_lefty")
    print(lefty_min)
    #rightx = nonzerox[right_lane_inds]
    #righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    #画出所有检测到的点
    points_image=np.zeros((720,1280,3),np.float32)

    for left,right in zip(leftx,lefty):
        cv2.circle(points_image,(left,right),1,(0,255,0))
    cv2.imshow("points_img",points_image)
   
    #画出曲线
    left_fit = np.polyfit(leftx, lefty, 2)
    
    #right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, left_lane_inds,leftx_max

def draw_area(undist,binary_warped,Minv,left_fit,leftx_max):
    print(undist.shape)
    # Generate x and y values for plotting
    plotx = np.linspace(0, leftx_max-1, leftx_max )

    left_fity = left_fit[0]*plotx**2 + left_fit[1]*plotx + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #return curvature of the polyfit
    depth_y=360
    if ( 200<(-left_fit[1]+(left_fit[1]**2-4*left_fit[0]*(left_fit[2]-depth_y))**(1/2))/(2*left_fit[0])<1000 ):
        depth_x=(-left_fit[1]+(left_fit[1]**2-4*left_fit[0]*left_fit[2])**(1/2))/(2*left_fit[0])
    else:
        depth_x=(-left_fit[1]-(left_fit[1]**2-4*left_fit[0]*(left_fit[2]-depth_y))**(1/2))/(2*left_fit[0])
    curvature=2*left_fit[0]*depth_x+left_fit[1]

    # Create an image to draw the lines on
    color_warp=np.zeros((720,1280,3),dtype="uint8")
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([plotx, left_fity]))])
    #print(pts_left)
    #pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    #pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp,np.int32([pts_left]),False,(255,0,255),5)
    cv2.imshow("left_polyfit",color_warp)

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
  
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result,curvature,depth_x
#----------------------------主程序声明----------------------------
#程序读取
src_image=cv2.imread("D:\\OpenCV\\Project\\projectTest\\openCV\\AdvancedLaneDetection1.1\\ZED\\14.png")
cv2.imshow('image',src_image)
dst_image=src_image.copy()
#去除雷达区域
src_image[550:700,530:830]=(0,0,0)
#阈值过滤
threshold_image=thresholding(src_image)
cv2.imshow("threshold",threshold_image)
#鸟瞰视图
Minv=aerial_view_inverse(threshold_image)
bird_image=aerial_view(threshold_image)
cv2.imshow("bird_image",bird_image)
#查找车道线
left_fit,left_lane_inds,leftx_max=find_line(bird_image)
#绘制车道线
result,curvature,depth_x=draw_area(dst_image,bird_image,Minv,left_fit,leftx_max)
print(curvature,depth_x)
#结果显示
cv2.imshow("test2",result)
cv2.waitKey(0)
