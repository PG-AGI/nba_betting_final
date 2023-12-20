#!/bin/bash

# Wait for the database to be ready and initialize it
python initialize_db.py

# Then start the main application
exec "$@"
