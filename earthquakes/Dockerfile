FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY ./requirements.txt .

RUN pip install -r requirements.txt

# Set the user to the current user
ARG USER
ARG USER_ID
ARG GROUP_ID
RUN groupadd -g ${GROUP_ID} ${USER} && \
    useradd -m -u ${USER_ID} -g ${USER} ${USER}
RUN mkdir /home/${USER}/earthquakes
RUN chown ${USER_ID} /home/${USER}/earthquakes
RUN usermod --shell /bin/bash ${USER}
USER ${USER}
