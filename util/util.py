'''
utility functions assisting nuclei detection and segmentation
@author: Kemeng Chen
'''
from re import S
import numpy as np 
import pprint
import cv2
import os
import sys
import math
from time import time, ctime
from skimage.morphology import square, erosion, dilation
from skimage.measure import label, regionprops
import skimage
from .run_restored_model import restored_model

def print_ctime():
	current_time=ctime(int(time()))
	print(str(current_time))

def batch2list(batch):
	mask_list=list()
	for index in range(batch.shape[0]):
		mask_list.append(batch[index,:,:])
	return mask_list

def patch2image(patch_list, patch_size, stride, shape):	
	if shape[0]<patch_size:
		L=0
	else:
		L=math.ceil((shape[0]-patch_size)/stride)
	if shape[1]<patch_size:
		W=0
	else:
		W=math.ceil((shape[1]-patch_size)/stride)	

	full_image=np.zeros([L*stride+patch_size, W*stride+patch_size])
	bk=np.zeros([L*stride+patch_size, W*stride+patch_size])
	cnt=0
	for i in range(L+1):
		for j in range(W+1):
			full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]+=patch_list[cnt]
			bk[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]+=np.ones([patch_size, patch_size])
			cnt+=1   
	full_image/=bk
	image=full_image[0:shape[0], 0:shape[1]]
	# cv2.namedWindow("image")     # 创建一个image的窗口
	# cv2.imshow("image", image)    # 显示图像
	# cv2.waitKey(5)               # 默认为0，无限等待
	# cv2.destroyAllWindows()      # 释放所有窗口
	return image

def image2patch(in_image, patch_size, stride, blur=False, f_size=9):
	if blur is True:
		in_image=cv2.medianBlur(in_image, f_size)
		# in_image=denoise_bilateral(in_image.astype(np.float), 19, 11, 9, multichannel=False)
	shape=in_image.shape
	if shape[0]<patch_size:
		L=0
	else:
		L=math.ceil((shape[0]-patch_size)/stride)
	if shape[1]<patch_size:
		W=0
	else:
		W=math.ceil((shape[1]-patch_size)/stride)	
	patch_list=list()
	
	if len(shape)>2:
		full_image=np.pad(in_image, ((0, patch_size+stride*L-shape[0]), (0, patch_size+stride*W-shape[1]), (0,0)), mode='symmetric')
	else:
		full_image=np.pad(in_image, ((0, patch_size+stride*L-shape[0]), (0, patch_size+stride*W-shape[1])), mode='symmetric')
	for i in range(L+1):
		for j in range(W+1):
			if len(shape)>2:
				patch_list.append(full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size, :])
			else:
				patch_list.append(full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size])
	if len(patch_list)!=(L+1)*(W+1):
		raise ValueError('Patch_list: ', str(len(patch_list), ' L: ', str(L), ' W: ', str(W)))
	
	return patch_list

def list2batch(patches):
	'''
	covert patch to flat batch
	args:
		patches: list
	return:
		batch: numpy array
	'''
	patch_shape=list(patches[0].shape)

	batch_size=len(patches)
	
	if len(patch_shape)>2:
		batch=np.zeros([batch_size]+patch_shape)
		for index, temp in enumerate(patches):
			batch[index,:,:,:]=temp
	else:
		batch=np.zeros([batch_size]+patch_shape+[1])
		for index, temp in enumerate(patches):
			batch[index,:,:,:]=np.expand_dims(temp, axis=-1) # 通过在指定位置插入新的轴来扩展数组形状。
	return batch

def preprocess(input_image, patch_size, stride, file_path):
	f_size=5
	g_size=10
	# shape[0] =图像的高
	# shape[1] =图像的宽
	# shape[2] = 图像的图像通道数量
	shape=input_image.shape
	patch_list=image2patch(input_image, patch_size, stride)
	num_group=math.ceil(len(patch_list)/g_size)
	batch_group=list()
	for i in range(num_group):
		temp_batch=list2batch(patch_list[i*g_size:(i+1)*g_size])
		batch_group.append(temp_batch)
	return batch_group, shape

def sess_interference(sess, batch_group):
	patch_list=list()
	for temp_batch in batch_group:
		mask_batch=sess.run_sess(temp_batch)[0]
		mask_batch=np.squeeze(mask_batch, axis=-1)
		mask_list=batch2list(mask_batch)
		patch_list+=mask_list
	return patch_list

def center_point(mask):
	v,h=mask.shape
	center_mask=np.zeros([v,h])

	mask=erosion(mask, square(3))
	individual_mask=label(mask, connectivity=2)
	prop=regionprops(individual_mask)
	for cordinates in prop:
		temp_center=cordinates.centroid # 中心坐标
		print("temp_center",temp_center)
		if not math.isnan(temp_center[0]) and not math.isnan(temp_center[1]):
			temp_mask=np.zeros([v,h])
			temp_mask[int(temp_center[0]), int(temp_center[1])]=1
			center_mask+=dilation(temp_mask, square(2))
	# skimage.io.imshow(center_mask)
	# skimage.io.show()
	return np.clip(center_mask, a_min=0, a_max=1).astype(np.uint8)

def draw_individual_edge(mask,image):
	v,h=mask.shape
	cellCount = 0
	edge=np.zeros([v,h])
	individual_mask=label(mask, connectivity=2)
	for index in np.unique(individual_mask):
		if index==0:
			continue
		temp_mask=np.copy(individual_mask)
		temp_mask[temp_mask!=index]=0
		temp_mask[temp_mask==index]=1
		temp_mask=dilation(temp_mask, square(3))
		temp_edge=cv2.Canny(temp_mask.astype(np.uint8), 2,5)/255 # 边缘轮廓图
		temp_edge_copy = np.copy(temp_edge)
		image_copy = np.copy(image)
		r =  tailor_cell(temp_edge_copy,image_copy)
		edge+=temp_edge
		if r is None:
			# 表示面积不符合细胞规则不是细胞
			continue
		r = cv2.resize(r,(200, 200))
		cv2.imwrite(os.path.join("data/sample_1/cells", 'clll'+str(index)+".png"), r)
		cellCount +=1
		print("细胞数量->",cellCount)
	print("识别细胞数量(未筛选前)->",np.unique(individual_mask).size)
	return np.clip(edge, a_min=0, a_max=1).astype(np.uint8)

# 
def center_edge(mask, image):
	center_map=center_point(mask)
	edge_map=draw_individual_edge(mask,image)
	comb_mask=center_map+edge_map
	comb_mask=np.clip(comb_mask, a_min=0, a_max=1)
	check_image=np.copy(image)
	comb_mask*=255
	check_image[:,:,1]=np.maximum(check_image[:,:,1], comb_mask)
	return check_image.astype(np.uint8), comb_mask.astype(np.uint8)


def tailor_cell(img,sampleImage):
	img = img.astype(np.uint8)
	img1 = np.copy(img)
	img2 = np.copy(img)
	# print(img1.shape)
	# img = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)

	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	areas = []

	for c in range(len(contours)):
		areas.append(cv2.contourArea(contours[c]))

	max_id = areas.index(max(areas))
	area = cv2.contourArea(contours[max_id])
	print("面积->",area)
	if area<150:
		return
	elif area >1000:
		return
	max_rect = cv2.minAreaRect(contours[max_id])
	max_box = cv2.boxPoints(max_rect)
	max_box = np.int0(max_box)
	# img2 = cv2.drawContours(img2,[max_box],0,(0,255,0),2)

	pts1 = np.float32(max_box)
	# print("pts1",pts1)
	pts2 = np.float32([[max_rect[0][0]+max_rect[1][1]/2, max_rect[0][1]+max_rect[1][0]/2],
					[max_rect[0][0]-max_rect[1][1]/2, max_rect[0][1]+max_rect[1][0]/2],
					[max_rect[0][0]-max_rect[1][1]/2, max_rect[0][1]-max_rect[1][0]/2],
					[max_rect[0][0]+max_rect[1][1]/2, max_rect[0][1]-max_rect[1][0]/2]])
	pts2 = expansion(pts2,3,img1.shape)
			
	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img2, M, (img2.shape[1],img2.shape[0]))
	# 此处可以验证 max_box点的顺序
	color = [(0, 0, 255),(0,255,0),(255,0,0),(255,255,255)]
	i = 0
	for point in pts2:
		cv2.circle(dst, (int(point[0]),int(point[1])), 2, color[i], 4)
		i+=1
	# print("pts2",pts2)
	# 剪裁的图片
	target = sampleImage[int(pts2[2][1]):int(pts2[1][1]),int(pts2[2][0]):int(pts2[3][0])]
	imageVar = getImageVar(target)
	print("图片明暗度->",imageVar)
	# skimage.io.imshow(img1)
	# skimage.io.show()
	# skimage.io.imshow(target)
	# skimage.io.show() 
	return target

def expansion(pts2,num,shape):
	
	if(pts2[0][0]+num>shape[1]):
		pts2[0][0] = shape[1]
	else:
		pts2[0][0] = pts2[0][0]+num

	if(pts2[0][1]+num>shape[0]):
		pts2[0][1] = shape[0]
	else:
		pts2[0][1] = pts2[0][1]+num

	if(pts2[1][0]-num<0):
		pts2[1][0] = 1
	else:
		pts2[1][0] =pts2[1][0]-num

	if(pts2[1][1]+num>shape[0]):
		pts2[1][1] = shape[0]
	else:
		pts2[1][1] = pts2[1][1]+num

	if(pts2[2][0]-num<0):
		pts2[2][0] = 1
	else:
		pts2[2][0] = pts2[2][0] - num

	if(pts2[2][1]-num<0):
		pts2[2][1] = 1
	else:
		pts2[2][1] = pts2[2][1]-num

	if(pts2[3][0]+num>shape[1]):
		pts2[3][0] = shape[1]
	else:
		pts2[3][0] = pts2[3][0]+num

	if(pts2[3][1]-num<0):
		pts2[3][1] = 1
	else:
		pts2[3][1] = pts2[3][1]-num
	return pts2

def getImageVar(img):
  image = np.copy(img).astype(np.uint8)
  img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
  return imageVar