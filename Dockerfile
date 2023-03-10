FROM python:3.10
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY ./Data ../Data
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py", "--server.headless", "true"]