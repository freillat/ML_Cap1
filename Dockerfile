FROM public.ecr.aws/lambda/python:3.10
RUN pip install --upgrade pip
RUN pip install numpy==1.23.1
RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install pillow

COPY model_2025_satellite.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]