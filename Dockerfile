FROM python:3.10

WORKDIR /service_code

COPY  ./requirements.txt /service_code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /service_code/requirements.txt

COPY ./data /service_code/data
COPY ./final_model.tflite /service_code/final_model.tflite
COPY image_sorting_service.py /service_code/image_sorting_service.py

CMD ["uvicorn", "image_sorting_service:app", "--host", "0.0.0.0", "--port", "8000","--workers","1"]