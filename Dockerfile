FROM python:3.9-alpine AS wheel
WORKDIR /app
RUN apk add --update --no-cache build-base libffi-dev
COPY pyproject.toml poetry.lock /app/
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir poetry && \
  poetry config virtualenvs.create false && \
  poetry install --no-dev
COPY . /app/
RUN poetry build -f wheel

FROM python:3.9-alpine
RUN apk add --update --no-cache ffmpeg build-base libffi-dev
COPY --from=wheel /app/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl
RUN rm -rf /app

ENTRYPOINT [ "meetingscribe" ]
CMD [ "--help" ]
