FROM mambaorg/micromamba:0.15.3

EXPOSE 8501

USER root

RUN mkdir /opt/streamlit-fastapi-deploy-tutorial
RUN chmod -R 777 /opt/streamlit-fastapi-deploy-tutorial
WORKDIR /opt/streamlit-fastapi-deploy-tutorial


USER micromamba

COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
   micromamba clean --all --yes


USER root

# OG FASTAPI STUFF
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/opt/streamlit-fastapi-deploy-tutorial/jobs_classifier"

# END FASTAPI STUFF

# COPY ./jobs_classifier /opt/streamlit-fastapi-deploy-tutorial/jobs_classifier
# COPY ./pickle_files /opt/streamlit-fastapi-deploy-tutorial/pickle_files

COPY . streamlit-fastapi-deploy-tutorial/

COPY run.sh run.sh
RUN chmod a+x run.sh
CMD ["./run.sh"]
