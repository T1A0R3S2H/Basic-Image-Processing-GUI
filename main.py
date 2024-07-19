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
current_shape = None
start_x, start_y = None, None

# Open image function
def open_image():
    global original_image, processed_image
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path)
        processed_image = original_image.copy()
        display_image(original_image, left_canvas)
        display_image(processed_image, right_canvas)

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

# Shape drawing functions
def start_draw(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y

def draw_shape(event):
    global processed_image, start_x, start_y
    if original_image is not None and current_shape:
        end_x, end_y = event.x, event.y
        temp_image = processed_image.copy()
        if current_shape == "line":
            cv2.line(temp_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        elif current_shape == "rectangle":
            cv2.rectangle(temp_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        elif current_shape == "circle":
            radius = int(((end_x - start_x)**2 + (end_y - start_y)**2)**0.5)
            cv2.circle(temp_image, (start_x, start_y), radius, (0, 0, 255), 2)
        display_image(temp_image, right_canvas)

def end_draw(event):
    global processed_image, start_x, start_y
    if original_image is not None and current_shape:
        end_x, end_y = event.x, event.y
        if current_shape == "line":
            cv2.line(processed_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        elif current_shape == "rectangle":
            cv2.rectangle(processed_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        elif current_shape == "circle":
            radius = int(((end_x - start_x)**2 + (end_y - start_y)**2)**0.5)
            cv2.circle(processed_image, (start_x, start_y), radius, (0, 0, 255), 2)
        display_image(processed_image, right_canvas)

def select_shape(shape):
    global current_shape
    current_shape = shape
    status_bar.config(text=f"Selected shape: {shape}")

def deselect_shape(event):
    global current_shape
    canvas_width = left_canvas.winfo_width()
    canvas_height = left_canvas.winfo_height()
    if event.x < 0 or event.x > canvas_width or event.y < 0 or event.y > canvas_height:
        current_shape = None
        status_bar.config(text="Ready")

# Image processing operations
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
filter_menu = tk.Menu(menu_bar, tearoff=0)
filter_menu.add_command(label="Summation Filter", command=lambda: apply_filter("summation"))
filter_menu.add_command(label="Derivative Filter", command=lambda: apply_filter("derivative"))
edit_menu.add_cascade(label="Filters", menu=filter_menu)
menu_bar.add_cascade(label="Edit", menu=edit_menu)

# Add new Shape menu
shape_menu = tk.Menu(menu_bar, font=("Helvetica", 10), tearoff=0)
shape_menu.add_command(label="Line", command=lambda: select_shape("line"))
shape_menu.add_command(label="Rectangle", command=lambda: select_shape("rectangle"))
shape_menu.add_command(label="Circle", command=lambda: select_shape("circle"))
menu_bar.add_cascade(label="Shapes", menu=shape_menu)

root.config(menu=menu_bar)

# Create a status bar
status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

# Add keyboard shortcuts
root.bind("<Control-o>", lambda event: open_image())
root.bind("<Control-q>", lambda event: root.quit())

# Bind mouse events for drawing
left_canvas.bind("<ButtonPress-1>", start_draw)
left_canvas.bind("<B1-Motion>", draw_shape)
left_canvas.bind("<ButtonRelease-1>", end_draw)
left_canvas.bind("<Leave>", deselect_shape)

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
