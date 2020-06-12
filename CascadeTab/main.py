from border import border
from mmdet.apis import inference_detector, show_result, init_detector
import cv2
from Functions.blessFunc import borderless
import lxml.etree as etree
import glob


############ To Do ############
image_path = '/content/drive/My Drive/Optum/Dataset/images/*'
xmlPath = '/content/drive/My Drive/Optum/Dataset/XML file/'

config_fname = '/content/drive/My Drive/Optum/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py' 
checkpoint_path = '/content/drive/My Drive/Optum/Checkpoint/'
epoch = 'epoch_36.pth'
##############################

model = init_detector(config_fname, checkpoint_path+epoch)
# print (model)

# List of images in the image_path
imgs = glob.glob(image_path)
for i in imgs:

    print (i)
    img = cv2.imread(i)
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2] 

    print('Image Dimension    : ',dimensions)
    print('Image Height       : ',height)
    print('Image Width        : ',width)
    print('Number of Channels : ',channels) 
    
    result = inference_detector(model, i)
    print ("The result is = ",result)

    res_border = []
    res_bless = []
    res_cell = []

    root = etree.Element("document")

    #print ("[0][0]",result[0][0])
    #print ("[0][1]",result[0][1])
    #print ("[0][2]",result[0][2])

    ## for border
    for r in result[0][0]:
        print ("1.",r[4])
        if r[4]>.85:
            res_border.append(r[:4].astype(int))

    ## for cells
    for r in result[0][1]:
        print ("2.",r[4])
        if r[4]>.85:
            r[4] = r[4]*100
            res_cell.append(r.astype(int))

    ## for borderless
    for r in result[0][2]:
        print ("3.",r[4])
        if r[4]>.85:
            res_bless.append(r[:4].astype(int))

    print ("res_border",res_border)
    print ("res_cell",res_cell)
    print ("res_bless",res_bless)
    
    ## if border tables detected 
    if len(res_border) != 0:
        ## call border script for each table in image
        for res in res_border:
            try:
                root.append(border(res,cv2.imread(i)))  
            except:
                pass
    if len(res_bless) != 0:
        if len(res_cell) != 0:
            for no,res in enumerate(res_bless):
                root.append(borderless(res,cv2.imread(i),res_cell))

    myfile = open(xmlPath+i.split('/')[-1][:-3]+'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True,encoding="unicode"))
    myfile.close()