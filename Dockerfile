FROM public.ecr.aws/lambda/python:3.12
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
RUN pip install pillow

COPY model_2025_satellite.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]