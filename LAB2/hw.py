import numpy as np
import cv2

def split_image(image, threshold, max_depth=100):
    # Check if maximum recursion depth is reached
    if max_depth == 0:
        return [image]

    # Check if the image is homogeneous
    if np.std(image) < threshold:
        return [image]
    
    # If not homogeneous, split into four sub-images
    height, width, _ = image.shape
    half_height = height // 2
    half_width = width // 2
    
    sub_images = []
    sub_images.append(image[:half_height, :half_width])
    sub_images.append(image[:half_height, half_width:])
    sub_images.append(image[half_height:, :half_width])
    sub_images.append(image[half_height:, half_width:])
    
    # Recursively split each non-empty sub-image
    result = []
    for sub_image in sub_images:
        if sub_image.size != 0:  # Check for empty sub-image
            result.extend(split_image(sub_image, threshold, max_depth - 1))
    
    return result

# Read the input RGB image
image = cv2.imread("/Users/rakibulnasib/Desktop/image/LAB2/Lena.jpg")

# Set the threshold value (you can make it user-input if needed)
threshold = 30

# Split the image using the splitting technique
split_images = split_image(image, threshold)

# Visualize the split images
for i, split_image in enumerate(split_images):
    if split_image.size != 0:
        cv2.imshow(f"Split Image {i}", split_image)
        print(f"Split Image {i} size:", split_image.shape)
    else:
        print(f"Split Image {i} is empty")
 

# Wait for a key event and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
