FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY daruma_binding /app/daruma_binding
COPY sample_data /app/sample_data
COPY run_example.sh /app/run_example.sh
COPY README.md /app/README.md
COPY License /app/License

ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

ENTRYPOINT ["python3", "daruma_binding/predict.py"]
CMD ["--help"]