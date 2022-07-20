# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
# RUN pip3 install --no-cache-dir --upgrade pip
# RUN pip3 install --upgrade pip setuptools wheel
# RUN curl --proto '=https' --tlsv1.2 -sSf -y https://sh.rustup.rs | sh
# RUN rustc --version

# RUN pip3 install --no-cache-dir cython
# RUN pip3 install --no-cache-dir --upgrade wheel
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./jobs_classifier /code/jobs_classifier
COPY ./pickle_files /code/pickle_files

ENV PYTHONPATH "${PYTHONPATH}:/code/jobs_classifier"

# 
CMD ["uvicorn", "jobs_classifier.main:app", "--host", "0.0.0.0", "--port", "80"]
