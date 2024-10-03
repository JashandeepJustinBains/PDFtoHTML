import subprocess

# List of classes
classes_complete = ("30","31","32","33","34","35")
classes=("36","37","38","39",
        "41","42","43","44","45","46","47","48","49",
        "4a","4b","4c","4d","4e","4f",
        "50","51","52","53","54","55","56","57","58","59",
        "5a",
        "61","62","63","64","65","66","67","68","69",
        "6a","6b","6c","6d","6e","6f",
        "70","71","72","73","74","75","76","77","78","79",
        "7a")
# Path to the 3_load_data.py script
script_path = "F:/PDFtoHTML/3_load_data.py"

# Loop through each class and run the Python script
for class_name in classes:
    print(f"Processing class: {class_name}")
    process = subprocess.Popen(["python", script_path, class_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print the output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Print any remaining output
    rc = process.poll()
    if rc is not None:
        for output in process.stdout.readlines():
            print(output.strip())
        for error in process.stderr.readlines():
            print(error.strip())
    
    print(f"Finished processing class: {class_name}")
