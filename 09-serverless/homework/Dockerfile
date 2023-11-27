FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip3 install keras-image-helper
#RUN python3 -m pip install tflite-runtime
#RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
COPY ./tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl .
RUN pip3 install ./tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]

