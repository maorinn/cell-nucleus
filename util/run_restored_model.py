#coding :UTF-8
'''
restored a model and run session
@author: Kemeng Chen
'''
from keras.models import *
import skimage
import numpy as np 

class restored_model(object):

	def __init__(self, model_name, model_folder):
		# self.graph=tf.Graph()
		# self.sess=K.get_session()
		# K.set_session(self.sess)
		print('Read model: ', model_name)
		# with self.graph.as_default():
		self.model_saver=load_model(model_name)

	def run_sess(self, patches):
		# generated_mask=self.sess.run([self.c_mask_out], feed_dict)
		# print("patches->>",patches)
		skimage.io.imshow(patches[0].astype(np.uint8))
		skimage.io.show()
		print("patches,shape->>",patches[0].dtype)
		# img = resize(
		# 	img,
		# 	(IMG_HEIGHT, IMG_WIDTH),
		# 	mode='constant',
		# 	preserve_range=True
		# )
		generated_mask=self.model_saver.predict(patches)
		skimage.io.imshow(generated_mask[0])
		skimage.io.show()
		print('generated_mask-->',generated_mask)
		return generated_mask

	def close_sess(self):
		print()
		# self.sess.close()