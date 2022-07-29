#!/bin/bash
# tar -xzf /mnt/letsencrypt/etc.tar.gz -C / &&
# nginx -t &&
# service nginx start &&
# cron &&
# uvicorn streamlit-fastapi-deploy-tutorial/jobs_classifier.main:app --host 0.0.0.0 --port 3400 &&
streamlit run streamlit-fastapi-deploy-tutorial/jobs_classifier/app.py --theme.base "dark" &&
python streamlit-fastapi-deploy-tutorial/jobs_classifier/main.py