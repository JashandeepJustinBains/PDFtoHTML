import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageTk
import pyttsx3
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
import json
import cv2
import numpy as np

class TTSReader:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.stop_flag = False

    def highlight_text(self, page, word, highlight=True):
        rects = page.search_for(word)
        for rect in rects:
            if highlight:
                page.add_highlight_annot(rect)
            else:
                annots = page.annots()
                for annot in annots:
                    if annot.rect == rect:
                        annot.delete()

    def highlight_ocr_text(self, word):
        start_idx = ocr_text.search(word, "1.0", tk.END)
        if start_idx:
            end_idx = f"{start_idx}+{len(word)}c"
            ocr_text.tag_add("highlight", start_idx, end_idx)
            ocr_text.tag_config("highlight", background="yellow")
            ocr_text.see(start_idx)

    def read_aloud(self, text, page):
        self.stop_flag = False
        words = text.split()
        for word in words:
            if self.stop_flag:
                break
            self.highlight_text(page, word, highlight=True)
            self.engine.say(word)
            self.engine.runAndWait()
            self.highlight_text(page, word, highlight=False)
            self.highlight_ocr_text(word)
            if word.endswith('\n'):
                ocr_text.tag_remove("highlight", "1.0", tk.END)
            root.update_idletasks()  # Keep the GUI responsive
        update_tts_indicator(False)

    def start_reading(self, text, page):
        self.stop_flag = False
        update_tts_indicator(True)
        threading.Thread(target=self.read_aloud, args=(text, page)).start()

    def stop_reading(self):
        self.stop_flag = True
        self.engine.stop()
        update_tts_indicator(False)

    def set_speed(self, speed):
        self.engine.setProperty('rate', speed)

    def set_volume(self, volume):
        self.engine.setProperty('volume', float(volume) / 100)

# Initialize TTSReader instance
tts_reader = None
file_path = None
bookmarks = []

def read_from_cursor():
    global tts_reader
    if tts_reader is not None:
        tts_reader.stop_reading()
    tts_reader = TTSReader()
    cursor_index = ocr_text.index(tk.INSERT)
    text = ocr_text.get(cursor_index, tk.END)
    if doc:
        tts_reader.start_reading(text, doc.load_page(current_page))
    else:
        print(f"please chose a document")


def stop_tts():
    global tts_reader
    if tts_reader is not None:
        tts_reader.stop_reading()
        tts_reader = None

def set_tts_speed(speed):
    global tts_reader
    if tts_reader is not None:
        tts_reader.set_speed(speed)

def set_tts_volume(volume):
    global tts_reader
    if tts_reader is not None:
        tts_reader.set_volume(volume)

# Function to display the PDF page and OCR text
def display_page(page_num):
    global doc, current_page, tts_reader
    if tts_reader is not None:
        tts_reader.stop_reading()
        tts_reader = None
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = img.resize((800, int(800 * pix.height / pix.width)), Image.LANCZOS)  # Resize while maintaining aspect ratio
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




def open_image(file_path):
    global current_page, tts_reader, original_img, processed_img, display_original
    if tts_reader is not None:
        tts_reader.stop_reading()
        tts_reader = None

    img = Image.open(file_path)
    original_img = img.resize((800, int(800 * img.height / img.width)), Image.LANCZOS)  # Resize while maintaining aspect ratio
    img_tk = ImageTk.PhotoImage(original_img)

    # Display the original image in the GUI
    pdf_label.config(image=img_tk)
    pdf_label.image = img_tk

    # Convert the image to grayscale
    img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2GRAY)

    # Deskew the image gently
    coords = np.column_stack(np.where(img_cv > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_cv = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Apply adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply a gentle sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_thresh, -1, kernel)

    # Convert back to PIL image
    processed_img = Image.fromarray(cv2.cvtColor(img_sharp, cv2.COLOR_GRAY2RGB))

    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(img_sharp)
    if text:
        ocr_text.delete(1.0, tk.END)
        ocr_text.insert(tk.END, text)
    else:
        ocr_text.delete(1.0, tk.END)
        ocr_text.insert(tk.END, "No text found in the image.")

    # Update current page number
    current_page = 0
    page_entry.delete(0, tk.END)
    page_entry.insert(0, str(current_page + 1))
    total_pages_label.config(text="of 1")

    display_original = True  # Start by displaying the original image




def toggle_image():
    global display_original
    if display_original:
        img_tk = ImageTk.PhotoImage(processed_img)
        pdf_label.config(image=img_tk)
        pdf_label.image = img_tk
        display_original = False
    else:
        img_tk = ImageTk.PhotoImage(original_img)
        pdf_label.config(image=img_tk)
        pdf_label.image = img_tk
        display_original = True


def open_file():
    global doc, current_page, file_path, tts_reader
    file_path = filedialog.askopenfilename(filetypes=[("PDF and Image files", "*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("PDF files", "*.pdf"), ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if not file_path:
        return

    if tts_reader is not None:
        tts_reader.stop_reading()
        tts_reader = None

    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        current_page = 0
        display_page(current_page)
    else:
        open_image(file_path)

    save_state()  # Save the state after opening the file


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

# Function to update TTS status indicator
def update_tts_indicator(active):
    color = "green" if active else "red"
    tts_indicator.config(bg=color)
    speed_scale.config(state=tk.DISABLED if active else tk.NORMAL)
    volume_scale.config(state=tk.DISABLED if active else tk.NORMAL)

# Function to resize PDF viewer with window
def resize_pdf(event):
    clear_window()
    add_widgets()

# Function to clear the window
def clear_window():
    for widget in root.winfo_children():
        widget.grid_forget()

# Function to add widgets to the window
def add_widgets():
    button_frame.grid(row=0, column=0, columnspan=3, pady=5)

    open_button.grid(row=0, column=0, padx=5)
    start_button.grid(row=0, column=1, padx=5)
    stop_button.grid(row=0, column=2, padx=5)
    prev_button.grid(row=0, column=3, padx=5)
    next_button.grid(row=0, column=4, padx=5)
    page_label.grid(row=0, column=5, padx=5)
    page_entry.grid(row=0, column=6, padx=5)
    go_button.grid(row=0, column=7, padx=5)
    total_pages_label.grid(row=0, column=8, padx=5)
    speed_label.grid(row=0, column=9, padx=5)
    speed_scale.grid(row=0, column=10, padx=5)
    volume_label.grid(row=0, column=11, padx=5)
    volume_scale.grid(row=0, column=12, padx=5)
    tts_indicator.grid(row=0, column=13, padx=5)
    bookmark_button.grid(row=0, column=14, padx=5)

    pdf_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    pdf_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    ocr_text.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
    bookmarks_listbox.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

# Function to add a bookmark
def add_bookmark():
    global current_page
    if current_page not in bookmarks:
        bookmarks.append(current_page)
        bookmarks_listbox.insert(tk.END, f"Page {current_page + 1}")

# Function to go to a bookmark
def go_to_bookmark(event):
    global current_page
    selection = event.widget.curselection()
    if selection:
        index = selection[0]
        current_page = bookmarks[index]
        display_page(current_page)

# Function to save the current state
def save_state():
    state = {
        'file_path': file_path,
        'current_page': current_page,
        'cursor_index': ocr_text.index(tk.INSERT),
        'bookmarks': bookmarks
    }
    with open('state.json', 'w') as f:
        json.dump(state, f)

# Function to load the saved state
def load_state():
    global current_page, bookmarks, doc, file_path
    try:
        with open('state.json', 'r') as f:
            state = json.load(f)
            file_path = state['file_path']
            current_page = state['current_page']
            cursor_index = state['cursor_index']
            bookmarks = state['bookmarks']
            
            # Open the saved PDF file
            if file_path:
                doc = fitz.open(file_path)
                display_page(current_page)
                ocr_text.mark_set(tk.INSERT, cursor_index)
                for bookmark in bookmarks:
                    bookmarks_listbox.insert(tk.END, f"Page {bookmark + 1}")
            else:
                messagebox.showinfo("Info", "Please open a PDF file to restore the state.")
    except FileNotFoundError:
        pass
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load state: {e}")

def setup_gui():
    global root, button_frame, open_button, start_button, stop_button, prev_button, next_button
    global page_label, page_entry, go_button, total_pages_label, speed_label, speed_scale
    global volume_label, volume_scale, tts_indicator, bookmark_button, pdf_frame, pdf_label
    global ocr_text, bookmarks_listbox, toggle_button

    root = tk.Tk()
    root.title("Interactive PDF Reader")
    root.geometry("1200x800")
    root.bind("<Configure>", resize_pdf)

    button_frame = tk.Frame(root)
    button_frame.grid(row=0, column=0, columnspan=3, pady=5)

    open_button = tk.Button(button_frame, text="Open PDF", command=open_file)
    open_button.grid(row=0, column=0, padx=5)

    start_button = tk.Button(button_frame, text="Start Reading", command=read_from_cursor)
    start_button.grid(row=0, column=1, padx=5)

    stop_button = tk.Button(button_frame, text="Stop Reading", command=stop_tts)
    stop_button.grid(row=0, column=2, padx=5)

    prev_button = tk.Button(button_frame, text="Previous Page", command=prev_page)
    prev_button.grid(row=0, column=3, padx=5)

    next_button = tk.Button(button_frame, text="Next Page", command=next_page)
    next_button.grid(row=0, column=4, padx=5)

    page_label = tk.Label(button_frame, text="Page:")
    page_label.grid(row=0, column=5, padx=5)

    page_entry = tk.Entry(button_frame, width=5)
    page_entry.grid(row=0, column=6, padx=5)

    go_button = tk.Button(button_frame, text="Go", command=go_to_page)
    go_button.grid(row=0, column=7, padx=5)

    total_pages_label = tk.Label(button_frame, text="of 0")
    total_pages_label.grid(row=0, column=8, padx=5)

    speed_label = tk.Label(button_frame, text="TTS Speed:")
    speed_label.grid(row=0, column=9, padx=5)

    speed_scale = tk.Scale(button_frame, from_=100, to=500, orient=tk.HORIZONTAL, command=lambda val: set_tts_speed(int(val)))
    speed_scale.set(200)  # Default speed
    speed_scale.grid(row=0, column=10, padx=5)

    volume_label = tk.Label(button_frame, text="Volume:")
    volume_label.grid(row=0, column=11, padx=5)

    volume_scale = tk.Scale(button_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=lambda val: set_tts_volume(int(val)))
    volume_scale.set(100)  # Default volume
    volume_scale.grid(row=0, column=12, padx=5)

    tts_indicator = tk.Label(button_frame, text="TTS Status", width=10, bg="red")
    tts_indicator.grid(row=0, column=13, padx=5)

    bookmark_button = tk.Button(button_frame, text="Add Bookmark", command=add_bookmark)
    bookmark_button.grid(row=0, column=14, padx=5)

    toggle_button = tk.Button(button_frame, text="Toggle Image", command=toggle_image)
    toggle_button.grid(row=0, column=15, padx=5)

    pdf_frame = tk.Frame(root)
    pdf_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    pdf_label = tk.Label(pdf_frame)
    pdf_label.pack(fill=tk.BOTH, expand=True)

    ocr_text = tk.Text(root, wrap=tk.WORD, height=30, width=50, font=("Helvetica", 16))
    ocr_text.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    bookmarks_listbox = tk.Listbox(root)
    bookmarks_listbox.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
    bookmarks_listbox.bind('<<ListboxSelect>>', go_to_bookmark)

    # Configure grid weights to ensure proper resizing
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(1, weight=1)

    # Save state on close
    root.protocol("WM_DELETE_WINDOW", lambda: (save_state(), root.destroy()))

    # Load state on start
    load_state()

    root.mainloop()



def main():
    setup_gui()
if __name__ == "__main__":
    main()
