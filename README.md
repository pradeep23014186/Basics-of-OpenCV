# Basics-of-OpenCV
## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** G.Pradeep Kumar
- **Register Number:** 212223230150

## Ex. No. 01

### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_GRAYSCALE)
```

### 2. Print the image width, height & Channel.
```python
h, w = img_gray.shape
print(f"Width: {w}, Height: {h}, Channels: 1")
```

### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img_gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
```
<img width="552" height="452" alt="image" src="https://github.com/user-attachments/assets/47e2280c-0ffd-4c9d-8e2c-acd03d98f672" />


### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight_gray.png', img_gray)
```

### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
colour_img = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_COLOR)
colour_img = cv2.cvtColor(colour_img, cv2.COLOR_BGR2RGB)
```

### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(colour_img)
plt.title("Color Image")
plt.axis("off")
plt.show()

h, w, c = colour_img.shape
print(f"Width: {w}, Height: {h}, Channels: {c}")
```
<img width="561" height="483" alt="image" src="https://github.com/user-attachments/assets/0dce2eee-1448-450e-8f4e-50f102793e6a" />


### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
img_crop = colour_img[30:425, 200:550]
plt.imshow(img_crop)
plt.title("Cropped Image")
plt.axis("off")
plt.show()
```
<img width="401" height="450" alt="image" src="https://github.com/user-attachments/assets/8bbd5913-d1a2-41d2-8619-52cbe71a3806" />


### 8. Resize the image up by a factor of 2x.
```python
img_resized = cv2.resize(img_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(img_resized)
plt.title("Resized Image 2x")
plt.axis("off")
plt.show()
```
<img width="412" height="470" alt="image" src="https://github.com/user-attachments/assets/c3bf9406-b476-43bb-912f-c85b3a20b574" />


### 9. Flip the cropped/resized image horizontally.
```python
img_flip = cv2.flip(img_resized, 1)
plt.imshow(img_flip)
plt.title("Flipped Image")
plt.axis("off")
plt.show()
```
<img width="405" height="462" alt="image" src="https://github.com/user-attachments/assets/76843d57-0f5d-4a5e-95f1-ceb64a8e6409" />

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
apollo = cv2.imread('Apollo-11-launch.jpg')

plt.imshow(img2)
plt.title("Apollo")
plt.axis("on")
plt.show()
```
<img width="647" height="377" alt="image" src="https://github.com/user-attachments/assets/e4fb6da1-35d9-498e-8340-b3281e2bda6d" />

### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
font_color = (255, 255, 255)
img3 = apollo.copy()
img3 = cv2.putText(img3, text, (90, 700), font_face, 3.2, font_color, 4, cv2.LINE_AA)
```

### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = (255, 0, 255)
mag_img = img3.copy()
mag_img = cv2.rectangle(mag_img, (500, 50), (700, 650), rect_color, 10, cv2.LINE_8)
```

### 13. Display the final annotated image.
```python
plt.imshow(mag_img)
plt.title("IMAGE")
plt.axis("off")
plt.show()
```
<img width="583" height="354" alt="image" src="https://github.com/user-attachments/assets/ccb343cc-36a1-41b9-97d9-ec07f300c43a" />


### 14. Read the image ('Boy.jpg').
```python
boy = cv2.imread('Boy.jpg')
boy = cv2.cvtColor(boy, cv2.COLOR_BGR2RGB)
```

### 15. Adjust the brightness of the image.
```python
matrix_ones = np.ones(boy.shape, dtype="uint8") * 50
matrix_ones
```

### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(boy, matrix_ones)
img_darker = cv2.subtract(boy, matrix_ones)
```

### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(boy); ax[0].set_title("Original"); ax[0].axis("off")
ax[1].imshow(img_darker); ax[1].set_title("Darker"); ax[1].axis("off")
ax[2].imshow(img_brighter); ax[2].set_title("Brighter"); ax[2].axis("off")
plt.show()
```
<img width="828" height="226" alt="image" src="https://github.com/user-attachments/assets/72ca427d-691b-4902-b405-609db009e1f6" />


### 18. Modify the image contrast.
```python
img_higher1 = cv2.convertScaleAbs(boy, alpha=1.1, beta=0)
img_higher2 = cv2.convertScaleAbs(boy, alpha=1.2, beta=0)
```

### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(boy); ax[0].set_title("Original"); ax[0].axis("off")
ax[1].imshow(img_higher1); ax[1].set_title("Contrast 1.1x"); ax[1].axis("off")
ax[2].imshow(img_higher2); ax[2].set_title("Contrast 1.2x"); ax[2].axis("off")
plt.show()
```
<img width="831" height="214" alt="image" src="https://github.com/user-attachments/assets/fcc7e554-c288-4b48-b128-ca3f691beaf0" />

### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(boy)

plt.figure(figsize = (18, 5))
plt.subplot(141); plt.imshow(r); plt.title("BLUE CHANNEL")
plt.subplot(142); plt.imshow(g); plt.title("GREEN CHANNEL")
plt.subplot(143); plt.imshow(b); plt.title("RED CHANNEL");
```
<img width="1216" height="316" alt="image" src="https://github.com/user-attachments/assets/e1b77594-70e6-4fb8-af10-b9984f794c16" />

### 21. Merged the R, G, B , displays along with the original image
```python
merged_rgb = cv2.merge((r, g, b))

plt.figure(figsize = (22, 5))
plt.subplot(131); plt.imshow(boy); plt.title("ORIGINAL")
plt.subplot(132); plt.imshow(merged_rgb); plt.title("MERGED");
```
<img width="832" height="316" alt="image" src="https://github.com/user-attachments/assets/84be7845-5c67-4318-9bc2-ca6e2b17cfc1" />

### 22. Split the image into the H, S, V components & Display the channels.
```python
img_hsv = cv2.cvtColor(boy, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(img_hsv)

plt.figure(figsize = (18,5))
plt.subplot(141); plt.imshow(h); plt.title("H CHANNEL")
plt.subplot(142); plt.imshow(s); plt.title("S CHANNEL")
plt.subplot(143); plt.imshow(v); plt.title("V CHANNEL");
```
<img width="829" height="218" alt="image" src="https://github.com/user-attachments/assets/8cf743b1-dd36-478b-b4d9-91851fad53b9" />

### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.merge((h, s, v))

plt.figure(figsize = (22,5))
plt.subplot(131); plt.imshow(boy); plt.title("ORIGINAL")
plt.subplot(132); plt.imshow(merged_hsv); plt.title("MERGED");
```
<img width="835" height="315" alt="image" src="https://github.com/user-attachments/assets/85d8f956-bd27-4ecd-aa7c-9cff471c4a90" />

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.
