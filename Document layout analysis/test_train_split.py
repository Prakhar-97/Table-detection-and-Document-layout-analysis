import json
import sklearn
import os
import glob
import pprint
import shutil

img_source_dir = '/home/prakhar/Publaynet/Original_data'
train = '/home/prakhar/Publaynet/train'
val = '/home/prakhar/Publaynet/validation'
subdirs = []
ratio = 0.7
for subdir in os.listdir(img_source_dir):

    print (subdir)
    a = os.path.join(img_source_dir, subdir)
    subdirs.append(a)

print (subdirs)

elements = len(subdirs)
middle = int(elements*ratio)

train_list = subdirs[:middle]
val_list = subdirs[middle:]

for f in train_list:
    shutil.move(f, train)

for f in val_list:
    shutil.move(f, val)


train_path = '/home/prakhar/Publaynet/train/*'
val_path = '/home/prakhar/Publaynet/validation/*'

train_imgs = glob.glob(train_path)
#print (train_imgs)
val_imgs = glob.glob(val_path)
#print (val_imgs)

with open('/home/prakhar/Publaynet/Labels/val.json') as f:
	data = json.load(f)

#pp = pprint.PrettyPrinter(indent=4)									
#pp.pprint (data)

def create_dict(imgs, data):

	train_ann = []
	name = []
	super_dict = {}
	total = len(imgs)

	for count,i in enumerate(imgs):

		print("Progress : ",count,"/",total)	
		image_name = os.path.split(i)
		name_list = data["images"]

		for j in name_list:

			if j["file_name"] == image_name[1]:

				num = j["id"]
				ann = data["annotations"]
				name.append(j)

				for k in ann:

					if k["image_id"] == num:

						train_ann.append(k)
				
	super_dict["annotations"] = train_ann
	super_dict["images"] = name
	super_dict["categories"] = data["categories"]
	
	return (super_dict)

print ("For train")
train_dict = create_dict(train_imgs, data)
pp = pprint.PrettyPrinter(indent=4)									
pp.pprint (train_dict)
with open("train_publaynet.json", 'w') as outfile:
	json.dump(train_dict, outfile)

print ("For val")
val_dict = create_dict(val_imgs, data)
pp = pprint.PrettyPrinter(indent=4)									
pp.pprint (val_dict)
with open("val_publaynet.json", 'w') as f:
	json.dump(val_dict, f)


