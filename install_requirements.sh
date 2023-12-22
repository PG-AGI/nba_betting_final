#!/bin/bash
set -e

install_package() {
    package=$1
    echo "Installing $package in the background..."
    pip install "$package" &
}

# Install each package in the background
while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    install_package "$requirement"
done < requirements.txt

# Wait for all background processes to finish
wait

echo "All packages installed."
