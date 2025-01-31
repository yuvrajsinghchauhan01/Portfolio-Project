# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app


# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK punkt_tab data
RUN python -m nltk.downloader punkt_tab

# Copy the rest of the application code
COPY . .



# Command to run the application using gunicorn
CMD ["python3","-m","flask","run","--host=0.0.0.0"]