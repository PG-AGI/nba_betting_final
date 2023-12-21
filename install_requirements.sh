#!/bin/bash
set -e

# Function to install a package and handle potential errors
install_package() {
    package=$1
    echo "Attempting to install $package..."
    if pip install "$package"; then
        echo "$package installed successfully."
    else
        echo "Failed to install $package. Skipping..."
    fi
}

# Read each line in requirements.txt and attempt to install
while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    install_package "$requirement"
done < requirements.txt
