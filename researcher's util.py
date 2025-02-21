from PIL import Image
import os
import shutil
import numpy as np
import cv2
import random

def overlay(save_dir, image_path1, image_path2, image_path3=None):
    # Load the two images
    img1 = Image.open(image_path1).convert("RGBA")
    img2 = Image.open(image_path2).convert("RGBA")
    if(image_path3 != None):
        img3 = Image.open(image_path3).convert("RGBA")

    # Resize img2 to match img1 size
    img2 = img2.resize(img1.size)
    if(image_path3 != None):
        img3 = img3.resize(img1.size)

    # Blend images (you can adjust the alpha for transparency)
    blended_image = Image.blend(img1, img2, 0.5)
    if(image_path3 != None):
        blended_image = Image.blend(blended_image, img3, 0.33)

    if(not os.path.exists(save_dir[0])): os.makedirs(save_dir[0])
    blended_image.save(os.path.join(save_dir[0], save_dir[1]))

def overlay_master():
    for dirpath, dirnames, filenames in os.walk('.'):
        if(sorted(dirnames) == ['t1c', 't1n', 't2f', 't2w'] or sorted(dirnames) == ['eht', 'ntc', 'sfh', 't1c', 't1n', 't2f', 't2w']): 
            #if(os.path.exists(os.path.join(dirpath, 'et'))): shutil.rmtree(os.path.join(dirpath, 'et'))
            #if(os.path.exists(os.path.join(dirpath, 'netc'))): shutil.rmtree(os.path.join(dirpath, 'netc'))
            #if(os.path.exists(os.path.join(dirpath, 'snfh'))): shutil.rmtree(os.path.join(dirpath, 'snfh'))
            if os.path.exists(os.path.join(dirpath, 'eht')): shutil.rmtree(os.path.join(dirpath, 'eht'))
            else: os.makedirs(os.path.join(dirpath, 'eht'))
            if os.path.exists(os.path.join(dirpath, 'ntc')): shutil.rmtree(os.path.join(dirpath, 'ntc'))
            else: os.makedirs(os.path.join(dirpath, 'ntc'))
            if os.path.exists(os.path.join(dirpath, 'sfh')): shutil.rmtree(os.path.join(dirpath, 'sfh'))
            else: os.makedirs(os.path.join(dirpath, 'sfh'))

            for file in os.listdir(os.path.join(dirpath, 't2w')):
                if(not file.endswith('.png')): continue
                # Enhanced tumor
                t1c = os.path.join(dirpath, f't1c/{file[:-7]}t1c.png')
                t1n = os.path.join(dirpath, f't1n/{file[:-7]}t1n.png')
                t2w = os.path.join(dirpath, f't2w/{file[:-7]}t2w.png')
                t2f = os.path.join(dirpath, f't2f/{file[:-7]}t2f.png')

                overlay((os.path.join(dirpath, 'eht'), file[:-7]+'eht.png'), t1c, t2w)

                # Non-enhanced tumor core
                overlay((os.path.join(dirpath, 'ntc'), file[:-7]+'ntc.png'), t1c, t1c)

                # Surrounding FLAIR Hyperintensity
                overlay((os.path.join(dirpath, 'sfh'), file[:-7]+'sfh.png'), t2w, t2f)

def extract_same_modalities(modality):
    if(os.path.exists('./Brain')): shutil.rmtree('./Brain')

    if(not os.path.exists('./Brain/test/good/img')): os.makedirs('./Brain/test/good/img')
    if(not os.path.exists('./Brain/test/good/anomaly_mask')): os.makedirs('./Brain/test/good/anomaly_mask')
    if(not os.path.exists('./Brain/test/Ungood/img')): os.makedirs('./Brain/test/Ungood/img')
    if(not os.path.exists('./Brain/test/Ungood/anomaly_mask')): os.makedirs('./Brain/test/Ungood/anomaly_mask')
    
    if(not os.path.exists('./Brain/valid/good/img')): os.makedirs('./Brain/valid/good/img')
    if(not os.path.exists('./Brain/valid/good/anomaly_mask')): os.makedirs('./Brain/valid/good/anomaly_mask')
    if(not os.path.exists('./Brain/valid/Ungood/img')): os.makedirs('./Brain/valid/Ungood/img')
    if(not os.path.exists('./Brain/valid/Ungood/anomaly_mask')): os.makedirs('./Brain/valid/Ungood/anomaly_mask')

    for dirpath, dirnames, filenames in os.walk('.'):
        if(sorted(dirnames) == ['t1c', 't1n', 't2f', 't2w'] or sorted(dirnames) == ['eht', 'ntc', 'sfh', 't1c', 't1n', 't2f', 't2w']): 
            for dir in [modality]:#dirnames:
                under_path = os.path.join(dirpath, dir)
                for file in os.listdir(under_path):
                    #print(dirpath)
                    shutil.copy(os.path.join(under_path, file), 'Brain/'+'/'.join(dirpath.split('\\')[2:]))
        elif(sorted(dirnames) == ['anomaly_mask', 'img']):
            dir = 'anomaly_mask'
            under_path = os.path.join(dirpath, dir)
            for file in os.listdir(under_path):
                shutil.copy(os.path.join(under_path, file), os.path.join('Brain','/'.join(dirpath.split('\\')[2:]), dir, file[:-7] + modality + '.png'))

def ground_truth_separator_master(start):
    for dirpath, dirnames, filenames in os.walk(start):
        if (dirpath == r'./Brain\test\Ungood\anomaly_mask' or dirpath == r'./Brain\valid\Ungood\anomaly_mask'):
            for file in filenames:
                ground_truth_separator(os.path.join(dirpath, file), mode=file[-7:-4])

def ground_truth_separator(file_path, mode='eht'):
    a_start = Image.open(file_path).convert('L')
    a = np.array(a_start)

    a_start.close()

    # interchangeable
    thresholds = [85, 170, 255]

    # for row_i, row in enumerate(a):
    #     for col_i, pixel in enumerate(row):
    #         if pixel > 10:
    #             a[row_i][col_i] = thresholds[min([(i, abs(pixel - val)) for i, val in enumerate(thresholds)], key= lambda x: x[1])[0]]
    
    #print(np.max(a))
    if mode == 'eht':
        a = np.where(a == 255, 255, 0)
    elif mode == 'ntc':
        a = np.where(a == 85, 255, 0)
    else:
        a = np.where(a == 170, 255, 0)

    a = Image.fromarray(a).convert('L')
    a.save(file_path)

def good_image_filter_master(start):
        for sub_dir in ['test', 'valid']:
            dirpath = os.path.join(start, sub_dir)
            
            cur_dir = os.path.join(dirpath, 'Ungood')
            dest_dir = os.path.join(dirpath, 'good')

            for file in os.listdir(os.path.join(cur_dir, 'anomaly_mask')):
                good_image_filter(os.path.join(cur_dir, 'anomaly_mask', file), os.path.join(cur_dir, 'img', file), os.path.join(dest_dir, 'img'))

        
def good_image_filter(anomaly, img, dest):

        image = cv2.imread(anomaly, cv2.IMREAD_GRAYSCALE)  # You can use IMREAD_GRAYSCALE for grayscale
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        flag = False
        if np.max(image_array) == 0 or np.count_nonzero(image_array == np.max(image_array)) <= 100:
            flag = True

        if(flag == True): # images with anomalies
            #send OTHER files to other destination
            shutil.move(img, dest)
            shutil.move(anomaly, os.path.join(os.path.join(dest, os.pardir), 'anomaly_mask'))

def combine_tumor_comp():
    a = zip(sorted(os.listdir('ET')), sorted(os.listdir('NETC')), sorted(os.listdir('SNFH')))
    #print(list(os.listdir('ET')))
    for image in a:
        et = np.array(cv2.imread(os.path.join('ET', image[0]), cv2.IMREAD_GRAYSCALE))
        netc = np.array(cv2.imread(os.path.join('NETC', image[1]), cv2.IMREAD_GRAYSCALE))
        snfh = np.array(cv2.imread(os.path.join('SNFH', image[2]), cv2.IMREAD_GRAYSCALE))
        
        combined = np.where(netc == 255, 85, np.where(et == 255, 255, np.where(snfh == 255, 170, 0)))
        sav = Image.fromarray(combined).convert('L')

        if(not os.path.exists('anomaly_mask')): os.makedirs('anomaly_mask')
        sav.save(os.path.join('anomaly_mask', image[0][:-8]+'.png'))


def final_good_image_filter(anomaly, dest):

        image = cv2.imread(anomaly, cv2.IMREAD_GRAYSCALE)  # You can use IMREAD_GRAYSCALE for grayscale
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        flag = False
        if np.max(image_array) == 0 or (np.count_nonzero(image_array == 255) <= 25 and np.count_nonzero(image_array == 170) <= 25 and np.count_nonzero(image_array == 85) <= 25):
            flag = True

        if(flag == True): # images without anomalies
            #send OTHER files to other destination
            shutil.move(anomaly, dest)
            #shutil.move(anomaly, os.path.join(os.path.join(dest, os.pardir), 'anomaly_mask'))

def color(dir):

    for image in os.listdir(dir):
        image_path = os.path.join(dir, image)

        image_arr = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        
        color_map = {
            0: (0, 0, 0),       # Black = None
            85: (0, 0, 255),    # Red = NETC
            170: (0, 255, 0),   # Green = SNFH
            255: (255, 0, 0)    # Blue = ET
        }

        color_image = np.zeros((image_arr.shape[0], image_arr.shape[1], 3), dtype=np.uint8)

        for gray_value, bgr_color in color_map.items():
            color_image[image_arr == gray_value] = bgr_color

        sav = Image.fromarray(color_image)
        sav.save(image_path)

def combined_image_filter(path):

    for anomaly in os.listdir(path):
        file_path = os.path.join(path, anomaly)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # You can use IMREAD_GRAYSCALE for grayscale
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        flag = False
        if np.max(image_array) == 0 or (np.count_nonzero(image_array == 85) + np.count_nonzero(image_array == 170) + np.count_nonzero(image_array == 255)) <= 50:
            flag = True

        if(flag == True): # images without anomalies
            #send OTHER files to other destination
            os.remove(file_path)

def split_train_val_test():
    # Define the paths
    import random
    source_folder = "anomaly_mask"
    train_folder = "train"
    test_folder = "test"
    val_folder = "val"

    # Create the train, test, and val folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Get all image filenames from the source folder
    images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Shuffle the images randomly
    random.shuffle(images)

    # Define split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Calculate the number of images for each split
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    # Split the images
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Function to copy images to their respective folders
    def copy_images(image_list, destination_folder):
        for image in image_list:
            shutil.copy(os.path.join(source_folder, image), os.path.join(destination_folder, image))

    # Copy the images to train, test, and val folders
    copy_images(train_images, train_folder)
    copy_images(val_images, val_folder)
    copy_images(test_images, test_folder)

    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Testing images: {len(test_images)}")

'''function main()'''
# overlay_master()
# extract_same_modalities('sfh') #select your modality
# ground_truth_separator_master('./Brain')
# shutil.rmtree(os.path.join('Brain', 'test', 'good', 'anomaly_mask'))
# os.makedirs(os.path.join('Brain', 'test', 'good', 'anomaly_mask'))
#good_image_filter_master('./Brain_AD')
#combine_tumor_comp()
#combined_image_filter('anomaly_mask')
#color('anomaly_mask')


#split_train_val_test()




# for file in os.listdir('combined_images'):
#     final_good_image_filter(os.path.join('combined_images', file), 'good')

# for index, file in enumerate(os.listdir('good')):
#     #print(file[:14]+'GOO'+file[17:])
#     #print(file[:14]+'GOO-'+f'{index:05d}'+file[23:])
#     os.rename(os.path.join('good', file), os.path.join('good', file[:14]+'GOO-'+f'{index:05d}'+file[23:]))


def sampling():
    # Define the paths
    source_folder = 'anomaly_mask'
    destination_folder = 'extra_GLI'

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Filter for images that contain "GLI" in their filename
    gli_images = [img for img in os.listdir(source_folder) if "GLI" in img]

    # Check if there are at least 100 images to move
    if len(gli_images) < 507:
        print("Not enough images containing 'GLI' in their filename.")
    else:
        # Randomly select 100 images from the filtered list
        selected_images = random.sample(gli_images, 507)

        # Move each selected image to the destination folder
        for image in selected_images:
            src_path = os.path.join(source_folder, image)
            dest_path = os.path.join(destination_folder, image)
            shutil.move(src_path, dest_path)

        print("100 images with 'GLI' in the filename have been moved successfully.")

def vertically_flip():
    import os
    from PIL import Image

    # Define the folder path
    folder_path = 'anomaly_mask'

    # Loop through each file in the folder
    idx = 830
    for filename in os.listdir(folder_path):
        # Check if "MET" is in the filename
        if "MET" in filename:
            file_path = os.path.join(folder_path, filename)
            
            # Open the image, flip it vertically, and save it
            with Image.open(file_path) as img:
                flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                flipped_img.save(os.path.join(folder_path, f'segmask-BraTS-MET-{idx:05d}-000.png'))
                idx += 1


    print("All 'MET' images have been vertically flipped.")

def extract_BRATS_anomalies():
    folder_list = ['GLI_training', 'MEN_training', 'MET_training']

    for folder in folder_list:

        file_path1 = os.path.join(folder, 'test', 'Ungood', 'anomaly_mask')
        for file in os.listdir(file_path1):
            shutil.copy(os.path.join(file_path1, file), os.path.join('original_brats_anomalies', 'segmask-' + file[:-8] + '.png'))

        file_path2 = os.path.join(folder, 'valid', 'Ungood', 'anomaly_mask')
        for file in os.listdir(file_path2):
            shutil.copy(os.path.join(file_path2, file), os.path.join('original_brats_anomalies', 'segmask-' + file[:-8] + '.png'))

extract_BRATS_anomalies()
color('original_brats_anomalies')
#sampling()
#vertically_flip()