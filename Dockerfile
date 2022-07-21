# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./pickle_files /code/pickle_files
COPY ./jobs_classifier /code/jobs_classifier

ENV PYTHONPATH "${PYTHONPATH}:/code/jobs_classifier"

# 
CMD ["uvicorn", "jobs_classifier.main:app", "--host", "0.0.0.0", "--port", "80"]
