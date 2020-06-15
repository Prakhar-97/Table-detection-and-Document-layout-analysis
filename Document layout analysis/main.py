from mmdet.apis import inference_detector, show_result_pyplot, init_detector
from mmdet.core import encode_mask_results, tensor2imgs
import cv2
import os 

###################################################  TO DO  ###################################################
image_pth = 'Give the image path'

config_fname = "Give the config file path "
checkpoint_path = 'Give the checkpoint file path'
epoch = 'epoch_6.pth'

#############################################################################################################

model = init_detector(config_fname, checkpoint_path+epoch)
img = cv2.imread(image_pth)

result = inference_detector(model, img)
#print ("The result is = ",result)

results = []
bbox_results, mask_results = result

res_text= []
res_title = []
res_list = []
res_table = []
res_figure = []
all_classes = []

#for text
for r in bbox_results[0]:
    if r[4]>.85:
      res_text.append(r[:4].astype(int))
   
print ("No. of paragraphs on the page are == ",len(res_text))
all_classes.append(res_text)

#for title
for r in bbox_results[1]:
    if r[4]>.85:
      res_title.append(r[:4].astype(int))

print ("No. of headers on the page are == ",len(res_title))
all_classes.append(res_title)

#for list
for r in bbox_results[2]:
    if r[4]>.85:
      res_list.append(r[:4].astype(int))

print ("No. of lists on the page are == ",len(res_list))
all_classes.append(res_list)

#for table
for r in bbox_results[3]:
    if r[4]>.85:
      res_table.append(r[:4].astype(int))

print ("No. of the tables on the page are == ",len(res_table))
all_classes.append(res_table)

#for figure
for r in bbox_results[4]:
    if r[4]>.85:
      res_figure.append(r[:4].astype(int))

print ("No. of figures on the page are == ",len(res_figure))
all_classes.append(res_figure)

im2 = img.copy()
for count, category in enumerate(all_classes):
    #print ("The no. of bbox in these classes are == ",len(category))
    im1 = img.copy()
    colors = [(55,255,20), (0,0,255), (132,240,255), (0,247,255), (2,2,105)]
    filename = ["paragraph_boxes.jpg", "header_boxes.jpg", "list_boxes.jpg", "tabel_boxes.jpg", "figure_boxes.jpg"]

    for box in category :
        #print (count)
        #print(colors[count]) 
        cv2.rectangle(im1, (box[0], box[1]), (box[2], box[3]), colors[count], 2)
        cv2.rectangle(im2, (box[0], box[1]), (box[2], box[3]), colors[count], 2)

    directory = '/content/drive/My Drive/results'
    os.chdir(directory)
    name = filename[count]
    #print (name)    
    cv2.imwrite(name, im1)

directory = '/content/drive/My Drive/results'
os.chdir(directory)
result_file = "all_annotations.jpg"
cv2.imwrite(result_file, im2)

encoded_mask_results = encode_mask_results(mask_results)
print ("Encoded mask results are ==  ",encoded_mask_results)
result = bbox_results, encoded_mask_results

results.append(result)