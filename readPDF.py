import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageTk
import pyttsx3
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()
stop_reading = False
current_page = 0
doc = None

# Function to highlight and unhighlight text
def highlight_text(page, word, highlight=True):
    rects = page.search_for(word)
    for rect in rects:
        if highlight:
            page.add_highlight_annot(rect)
        else:
            annots = page.annots()
            for annot in annots:
                if annot.rect == rect:
                    annot.delete()

# Function to read text aloud and highlight words
def read_aloud(text, page):
    global stop_reading
    words = text.split()
    for word in words:
        if stop_reading:
            break
        highlight_text(page, word, highlight=True)
        engine.say(word)
        engine.runAndWait()
        highlight_text(page, word, highlight=False)
    update_tts_indicator(False)

# Function to read text from the current cursor position
def read_from_cursor():
    global stop_reading
    stop_reading = False
    update_tts_indicator(True)
    cursor_index = ocr_text.index(tk.INSERT)
    text = ocr_text.get(cursor_index, tk.END)
    threading.Thread(target=read_aloud, args=(text, doc.load_page(current_page))).start()

# Function to display the PDF page and OCR text
def display_page(page_num):
    global doc, stop_reading, current_page
    stop_reading = True
    engine.stop()
    update_tts_indicator(False)

    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_tk = ImageTk.PhotoImage(img)

    # Display the PDF page in the GUI
    pdf_label.config(image=img_tk)
    pdf_label.image = img_tk

    text = pytesseract.image_to_string(img)
    ocr_text.delete(1.0, tk.END)
    ocr_text.insert(tk.END, text)

    # Update current page number
    current_page = page_num
    page_entry.delete(0, tk.END)
    page_entry.insert(0, str(current_page + 1))
    total_pages_label.config(text=f"of {len(doc)}")

# Function to open and process PDF
def open_pdf():
    global doc, current_page
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not file_path:
        return

    doc = fitz.open(file_path)
    current_page = 0
    display_page(current_page)

# Function to stop reading
def stop_tts():
    global stop_reading
    stop_reading = True
    engine.stop()
    update_tts_indicator(False)

# Function to go to the previous page
def prev_page():
    global current_page
    if current_page > 0:
        current_page -= 1
        display_page(current_page)

# Function to go to the next page
def next_page():
    global current_page
    if current_page < len(doc) - 1:
        current_page += 1
        display_page(current_page)

# Function to go to a specific page
def go_to_page():
    global current_page
    try:
        page_num = int(page_entry.get()) - 1
        if 0 <= page_num < len(doc):
            current_page = page_num
            display_page(current_page)
        else:
            messagebox.showerror("Error", "Invalid page number")
    except ValueError:
        messagebox.showerror("Error", "Invalid page number")

# Function to adjust TTS speed
def set_tts_speed(speed):
    engine.setProperty('rate', speed)

# Function to update TTS status indicator
def update_tts_indicator(active):
    color = "green" if active else "red"
    tts_indicator.config(bg=color)

# Create GUI
root = tk.Tk()
root.title("Interactive PDF Reader")

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

open_button = tk.Button(button_frame, text="Open PDF", command=open_pdf)
open_button.pack(side=tk.LEFT, padx=5)

start_button = tk.Button(button_frame, text="Start Reading", command=read_from_cursor)
start_button.pack(side=tk.LEFT, padx=5)

stop_button = tk.Button(button_frame, text="Stop Reading", command=stop_tts)
stop_button.pack(side=tk.LEFT, padx=5)

prev_button = tk.Button(button_frame, text="Previous Page", command=prev_page)
prev_button.pack(side=tk.LEFT, padx=5)

next_button = tk.Button(button_frame, text="Next Page", command=next_page)
next_button.pack(side=tk.LEFT, padx=5)

page_label = tk.Label(button_frame, text="Page:")
page_label.pack(side=tk.LEFT, padx=5)

page_entry = tk.Entry(button_frame, width=5)
page_entry.pack(side=tk.LEFT, padx=5)

go_button = tk.Button(button_frame, text="Go", command=go_to_page)
go_button.pack(side=tk.LEFT, padx=5)

total_pages_label = tk.Label(button_frame, text="of 0")
total_pages_label.pack(side=tk.LEFT, padx=5)

speed_label = tk.Label(button_frame, text="TTS Speed:")
speed_label.pack(side=tk.LEFT, padx=5)

speed_scale = tk.Scale(button_frame, from_=100, to=300, orient=tk.HORIZONTAL, command=lambda val: set_tts_speed(int(val)))
speed_scale.set(200)  # Default speed
speed_scale.pack(side=tk.LEFT, padx=5)

tts_indicator = tk.Label(button_frame, text="TTS Status", width=10, bg="red")
tts_indicator.pack(side=tk.LEFT, padx=5)

pdf_label = tk.Label(root)
pdf_label.pack(side=tk.LEFT, padx=10, pady=10)

ocr_text = tk.Text(root, wrap=tk.WORD, height=30, width=50)
ocr_text.pack(side=tk.RIGHT, padx=10, pady=10)

root.mainloop()
