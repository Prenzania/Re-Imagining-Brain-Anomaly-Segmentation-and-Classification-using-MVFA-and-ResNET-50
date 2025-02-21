#!/bin/bash

# Define the folder path
folder_path="MVFA-good"

# Loop through each matching file in the folder
for file in "$folder_path"/BraTS-GOO[0-2]-*.png; do
  # Check if file exists to avoid errors
  [ -e "$file" ] || continue
  
  # Generate the new file name by removing the digit after "GOO"
  new_file_name="${file/GOO?/GOO}"

  # Rename the file
  mv "$file" "$new_file_name"
done