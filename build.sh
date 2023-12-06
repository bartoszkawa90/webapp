#!/bin/bash

# Update pip
echo "Updating pip..."
python3.9 pip install -U pip

# Install dependencies

echo "Installing project dependencies..."
python3.9 -m pip install -r requirements.txt

pip3 install db-sqlite3
echo $PYTHONPATH
export PYTHONPATH="/usr/local/lib/python3.7/site-packages:$PYTHONPATH"

# Make migrations
echo "Making migrations..."
python3.9 manage.py makemigrations --noinput
python3.9 manage.py migrate --noinput

# Collect staticfiles
echo "Collect static..."
python3.9 manage.py collectstatic --noinput --clear

echo "Build process completed!"