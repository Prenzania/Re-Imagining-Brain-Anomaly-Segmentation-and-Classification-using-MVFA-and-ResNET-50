from PIL import Image
import os

def overlay(image_path1, image_path2, image_path3=None):
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
        blended_image = Image.blend(blended_image, img3, 0.5)

    blended_image.save('try.png')

path_t1c = './raw_data/MEN_training/valid/Ungood/img/t1c'
path_t2w = './raw_data/MEN_training/valid/Ungood/img/t2w'

files_t1c = [os.path.join(path_t1c, f) for f in os.listdir(path_t1c) if os.path.isfile(os.path.join(path_t1c, f)) and f.endswith('.png')]
files_t2w = [os.path.join(path_t2w, f) for f in os.listdir(path_t2w) if os.path.isfile(os.path.join(path_t2w, f)) and f.endswith('.png')]

files_t1c.sort()
files_t2w.sort()

print(files_t1c)
print()
print(files_t2w)
#for i in range(len(files_t1c)):
overlay(files_t1c[0], files_t2w[0])

