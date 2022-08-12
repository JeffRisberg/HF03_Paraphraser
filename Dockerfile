FROM python:3.10-slim

RUN pip3 install --no-cache-dir protobuf==3.20 torch transformers sentencepiece sacremoses flask flask-restful

RUN python -c \
        "from transformers import *; \
        model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase'); \
        tokenizer = PegasusTokenizerFast.from_pretrained('tuner007/pegasus_paraphrase');"

RUN mkdir -p /usr/src/app
COPY paraphraser/src/main/python/paraphraser.py /usr/src/app
WORKDIR /usr/src/app




# Expose port and create entrypoint.
EXPOSE 3333
ENTRYPOINT ["python"]
CMD ["paraphraser.py"]