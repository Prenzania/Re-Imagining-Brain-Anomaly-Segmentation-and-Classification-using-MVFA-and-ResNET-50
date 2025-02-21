from PIL import Image
import os
import shutil
import numpy as np
import cv2

et_test = "/home/medanomaly/MVFA-owen-eht/new_segmented_images/"
netc_test = "/home/medanomaly/MVFA-backup/new_segmented_images/"
snfh_test = "/home/medanomaly/MVFA-geli-sfh/new_segmented_images/"

def combine_tumor_comp():
    a = zip(sorted(os.listdir(et_test)), sorted(os.listdir(netc_test)), sorted(os.listdir(snfh_test)))
    for image in a:
        et = cv2.imread(os.path.join(et_test, image[0]), cv2.IMREAD_GRAYSCALE)
        netc = cv2.imread(os.path.join(netc_test, image[1]), cv2.IMREAD_GRAYSCALE)
        snfh = cv2.imread(os.path.join(snfh_test, image[2]), cv2.IMREAD_GRAYSCALE)

        combined = np.where(netc == 255, 85, np.where(et == 255, 255, np.where(snfh == 255, 170, 0)))

        # Calculate the Otsu's threshold
        _, otsu_thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate the percentage of white pixels
        white_pixels = np.sum(otsu_thresh == 255)
        total_pixels = otsu_thresh.size
        white_percentage = (white_pixels / total_pixels) * 100
        
        sav = Image.fromarray(combined).convert('L')

        # Determine the folder based on white pixel percentage
        if white_percentage < 5:  # Adjust this percentage as needed
            sav.save(os.path.join('anomaly_mask', image[0][:-8].replace('MEN', 'GOOD').replace('MET', 'GOOD').replace('GLI', 'GOOD') + '.png'))
        else:
            sav.save(os.path.join('Result', image[0][:-8]+'.png'))


'''function main()'''
combine_tumor_comp()