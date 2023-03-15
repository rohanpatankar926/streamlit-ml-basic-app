FROM python:3.8

RUN mkdir /streamlit_app

COPY . /streamlit_app

WORKDIR /streamlit_app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]