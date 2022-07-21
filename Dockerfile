# start by pulling the python image
FROM python:3
# switch working directory
WORKDIR /Bloc_5
# copy the requirements file into the image
ADD . /Bloc_5
# install the dependencies and packages in the requirements file
RUN pip install --upgrade pip && pip install --pre --upgrade -r requirements.txt
# configure the container to run in an executed manner
CMD ["python","app.py"]
