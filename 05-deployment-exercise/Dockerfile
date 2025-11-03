FROM agrigorev/zoomcamp-model:2025

# model file pipeline_v2.bin is already in /code in this base image
WORKDIR /code

# install runtime deps needed to unpickle + serve
RUN pip install --no-cache-dir scikit-learn==1.6.1 fastapi==0.115.0 uvicorn[standard]==0.30.6

# copy only your service
COPY predict.py .

EXPOSE 9696
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
