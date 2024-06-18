import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# Initialize the GUI
root = tk.Tk()
root.title("Image Processing App")

# Create a grid layout
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Create the canvas objects
left_canvas = tk.Canvas(root, width=500, height=600)
right_canvas = tk.Canvas(root, width=500, height=600)

# Add labels for the canvases
left_label = tk.Label(root, text="Original Image")
left_label.grid(row=0, column=0, padx=10, pady=10, sticky="n")

right_label = tk.Label(root, text="Processed Image")
right_label.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Place the canvas objects
left_canvas.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
right_canvas.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# Global variables
original_image = None
processed_image = None

# Open image function
def open_image():
    global original_image
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path)
        display_image(original_image, left_canvas)

# Display image function
def display_image(img, canvas):
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    
    img_height, img_width = img.shape[:2]

    # Calculate the resize ratio to fit the image within the canvas dimensions
    ratio = min(canvas_width / img_width, canvas_height / img_height)
    new_width, new_height = int(img_width * ratio), int(img_height * ratio)

    # Resize the image while maintaining the aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height))

    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    photo = ImageTk.PhotoImage(image=img_pil)

    # Calculate coordinates to center the image in the canvas
    x_center = (canvas_width - new_width) // 2
    y_center = (canvas_height - new_height) // 2

    canvas.create_image(x_center, y_center, anchor=tk.NW, image=photo)
    canvas.image = photo

# Image processing operations
def crop_image(event):
    global processed_image
    if original_image is not None:
        x, y = event.x, event.y
        roi = cv2.selectROI(original_image)
        cropped = original_image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        processed_image = cropped
        display_image(processed_image, right_canvas)

def crop_image_menu():
    global original_image, processed_image
    if original_image is not None:
        roi = cv2.selectROI("Select ROI", original_image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()  # Close the ROI selector window
        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            cropped = original_image[y:y+h, x:x+w]
            processed_image = cropped
            display_image(processed_image, right_canvas)

def blur_image():
    global processed_image
    if original_image is not None:
        blurred = cv2.GaussianBlur(original_image, (5, 5), 0)
        processed_image = blurred
        display_image(processed_image, right_canvas)

def unblur_image():
    global processed_image
    if original_image is not None:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(original_image, -1, kernel)
        processed_image = sharpened
        display_image(processed_image, right_canvas)

def transparency_meter():
    global processed_image
    if original_image is not None:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        transparency_level = np.sum(mask == 255) / (mask.shape[0] * mask.shape[1])
        print(f"Transparency level: {transparency_level * 100:.2f}%")
        processed_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        display_image(processed_image, right_canvas)

def erode_image():
    global processed_image
    if original_image is not None:
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(original_image, kernel, iterations=1)
        processed_image = eroded
        display_image(processed_image, right_canvas)

def dilate_image():
    global processed_image
    if original_image is not None:
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(original_image, kernel, iterations=1)
        processed_image = dilated
        display_image(processed_image, right_canvas)

def show_histogram():
    global processed_image
    if original_image is not None:
        import matplotlib.pyplot as plt
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([original_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()

def apply_filter(filter_type):
    global processed_image
    if original_image is not None:
        if filter_type == "summation":
            kernel = np.ones((5, 5), np.float32) / 25
            filtered = cv2.filter2D(original_image, -1, kernel)
        elif filter_type == "derivative":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(original_image, -1, kernel)
        processed_image = filtered
        display_image(processed_image, right_canvas)

# Create a button
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.grid(row=2, column=0, padx=10, pady=10)

# Style the menu
menu_bar = tk.Menu(root, font=("Helvetica", 10))
edit_menu = tk.Menu(menu_bar, font=("Helvetica", 10), tearoff=0)
edit_menu.add_command(label="Blur", command=blur_image)
edit_menu.add_command(label="Unblur", command=unblur_image)
edit_menu.add_command(label="Transparency Meter", command=transparency_meter)
edit_menu.add_command(label="Erode", command=erode_image)
edit_menu.add_command(label="Dilate", command=dilate_image)
edit_menu.add_command(label="Histogram", command=show_histogram)
# edit_menu.add_command(label="Crop", command=crop_image_menu)
filter_menu = tk.Menu(menu_bar, tearoff=0)
filter_menu.add_command(label="Summation Filter", command=lambda: apply_filter("summation"))
filter_menu.add_command(label="Derivative Filter", command=lambda: apply_filter("derivative"))
edit_menu.add_cascade(label="Filters", menu=filter_menu)
menu_bar.add_cascade(label="Edit", menu=edit_menu)
root.config(menu=menu_bar)

# Create a status bar
status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

# Add keyboard shortcuts
root.bind("<Control-o>", lambda event: open_image())
root.bind("<Control-q>", lambda event: root.quit())

# Ensure images are centered when the window is resized
def on_resize(event):
    if original_image is not None:
        display_image(original_image, left_canvas)
    if processed_image is not None:
        display_image(processed_image, right_canvas)

left_canvas.bind("<Configure>", on_resize)
right_canvas.bind("<Configure>", on_resize)

# Start the GUI event loop
root.mainloop()
