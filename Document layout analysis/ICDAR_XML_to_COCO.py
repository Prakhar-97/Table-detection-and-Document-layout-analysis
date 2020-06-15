from pdf2image import convert_from_path
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError,PDFSyntaxError)
from bs4 import BeautifulSoup as bs
import glob
import os
import cv2
import numpy as np
import pprint
import pickle
import json

pdf_path = '/home/prakhar/mmdetection/convert2pdf/all pdfs/*'
xml_path = '/home/prakhar/mmdetection/convert2pdf/all xmls/*' 
pdfs = glob.glob(pdf_path)
xmls = glob.glob(xml_path)
a = 1
b = 0
img_list = []
ann_list = []
cate_list = []
super_dict = {}
categories = {}

for i in pdfs:

	print (i)
	tail_pdf = os.path.split(i)
	name_pdf = os.path.splitext(tail_pdf[1])
	#print(tail_pdf[1])
	#print (name_pdf[0]) 

	for j in xmls :
		
		tail_xml = os.path.split(j)
		name_xml = os.path.splitext(tail_xml[1])
		#print(tail_xml[1])
		#print (name_xml[0]) 

		if (name_pdf[0]+"-reg" == name_xml[0]):
			pages = convert_from_path(i)

			for k, page in enumerate(pages):
				
				image = {}

				fname = name_pdf[0] + "_page_" + str(k+1) + ".png"
				print (j)

				content = []
				with open(j, 'r') as file:
					#import pdb ; pdb.set_trace()

					content = file.readlines()
					content ="".join(content)
					bs_content = bs(content, "lxml")

					table = bs_content.find_all("region")
					#print (table)
					coords = []

					for p in table:

						ann = {}
						masks = []
						num = p["page"]
						#print (num)
						#print (k+1)
						if num == str(k+1) :
							
							b = b+1
							length = len(p.contents)
							bbox = p.contents[length-2]

							x1 = int(bbox["x1"])
							#print (x1)
							y1 = int(bbox["y1"])
							w = int(bbox["x2"])-int(bbox["x1"])
							#print (w)
							h = int(bbox["y2"])-int(bbox["y1"])
							#print (h)

							coords = [x1, y1, w, h]
							mask = [x1, y1, x1, y1+h, x1+w, y1+h, x1+w, y1]
							masks.append(mask)

							ann["area"] = float(w*h)
							ann["bbox"] = coords
							ann["segmentation"] = masks
							ann["category_id"] = 1
							ann["image_id"] = a
							ann["id"] = b
							ann["iscrowd"] = 0
							ann["ignore"] = 0
							ann_list.append(ann)

					if len(coords) > 0 :
						
						page.save(fname, "PNG")
						
						img = cv2.imread(fname)
						dimensions = img.shape
						height = img.shape[0]
						width = img.shape[1]
						channels = img.shape[2] 

						#print('Image Dimension    : ',dimensions)
						#print('Image Height       : ',height)
						#print('Image Width        : ',width)
						#print('Number of Channels : ',channels) 

						image["file_name"] = fname
						image["width"] = width
						image["height"] = height
						image["id"] = a
						
						img_list.append(image)
						a = a+1

categories["id"] = 1
categories["name"] = "table"
cate_list.append(categories)

super_dict["annotations"] = ann_list
super_dict["categories"] = cate_list
super_dict["images"] = img_list
super_dict["type"] = "instances"

filename = 'dataset'
outfile = open(filename,'wb')
pickle.dump(super_dict, outfile)
outfile.close()

with open("dataset.json", 'w') as outfile:
	json.dump(super_dict, outfile)
#print (pickled_object)
#unpickled_object = pickle.load(open(filename, 'rb'))
#print (unpickled_object)

#a = CustomDataset(pickle.loads(pickled_object))
pp = pprint.PrettyPrinter(indent=4)									
pp.pprint (super_dict)
  