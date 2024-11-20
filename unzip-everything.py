import os
import subprocess
import platform

# Use os.path.join() to construct paths in a platform-independent way
canonical_folder = os.path.join(".", "canonical")
zip_files_txt = os.path.join(".", "derived_data", "zip-files.txt")

def has_extension(filename, ext):
    return filename.split(".")[-1] == ext

def all_files_of_interest():
    # Use platform-specific command to list files in a directory
    if platform.system() == "Windows":
        output = subprocess.check_output(f"dir /b /s \"{canonical_folder}\"", shell=True, text=True)
    else:
        output = subprocess.check_output(f"find \"{canonical_folder}\" -type f", shell=True, text=True)
    filenames = output.strip().split("\n")
    return [filename for filename in filenames if has_extension(filename, "zip") or has_extension(filename, "csv") or has_extension(filename, "xpt")]

def all_zip_files():
    with open(zip_files_txt) as f:
        return f.read().strip().split("\n")

def join_strings(strings, delim):
    return delim.join(strings)

def file_stem(filename):
    parts = filename.split(".")
    if len(parts) == 1:
        return filename
    else:
        parts.reverse()
        return join_strings(parts[1:], os.path.sep)

def unzip_all():
    zip_files = all_zip_files()
    for z in zip_files:
        print(f"Unzipping {z}")
        folder_name = file_stem(z)
        # Use platform-specific commands to create a directory and unzip a file
        if platform.system() == "Windows":
            subprocess.call(f"mkdir \"{folder_name}\"", shell=True)
            subprocess.call(f"powershell -Command \"Expand-Archive -Path \\\"{z}\\\" -DestinationPath \\\"{folder_name}\\\"\"", shell=True)
        else:
            subprocess.call(f"mkdir -p \"{folder_name}\"", shell=True)
            subprocess.call(f"unzip \"{z}\" -d \"{folder_name}\"", shell=True)

unzip_all()

# Use platform-specific command to create an empty file
if platform.system() == "Windows":
    subprocess.call("type NUL > derived_data\\unzipped-everything", shell=True)
else:
    subprocess.call("touch derived_data/unzipped-everything", shell=True)
