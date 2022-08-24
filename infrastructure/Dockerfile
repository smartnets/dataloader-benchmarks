FROM nvidia/cuda:11.2.1-base-ubuntu20.04
ARG USER="worker"
ARG UID="1002"
ARG INDRA="indra-0.0.5-py3-none-any.whl"
ENV DEBIAN_FRONTEND "noninteractive"

ENV PATH "$PATH:/usr/local"
ENV PYTHONPATH "$PYTHONPATH:/home/$USER/workspace"


RUN apt-get update && apt-get install -y htop vim tar net-tools tshark curl tcpdump python3.9-dev python3-pip python3-setuptools python3.9-distutils libturbojpeg-dev libopencv-dev tcpdump aria2 git  linux-tools-5.8.0-43-generic && ln -s /usr/bin/python3.9 /usr/bin/python

RUN useradd -u $UID -ms /bin/bash $USER \ 
    && mkdir -p /home/$USER \
    && mkdir -p /home/$USER/workspace \
    && mkdir -p /home/$USER/workspace/datasets \
    && mkdir -p /home/$USER/workspace/results \
    && chown -R $USER:$USER /home/$USER \
    && usermod -a -G video $USER \
    && groupadd pcap \
    && usermod -a -G pcap $USER \
    && chgrp pcap /usr/sbin/tcpdump \
    && setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump\ 
    && ln -s /usr/sbin/tcpdump /usr/bin/tcpdump \
    && groupadd wireshark \
    && usermod -a -G wireshark $USER

RUN  echo "Yes" | dpkg-reconfigure wireshark-common 


Run echo "a"

COPY infrastructure/install.sh /tmp/install.sh
RUN chmod +x /tmp/install.sh && /bin/bash /tmp/install.sh

RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY infrastructure/requirements/req.txt /tmp/req.txt
RUN python -m pip install -r /tmp/req.txt

RUN git clone https://github.com/pdollar/coco/ \
    && cd coco/PythonAPI \
    && make \ 
    && python setup.py install \
    && cd ../.. \
    && rm -r coco

COPY infrastructure/$INDRA  /tmp/$INDRA
RUN python -m pip install /tmp/$INDRA


WORKDIR /home/$USER/workspace
USER $USER
COPY --chown=$USER src /home/$USER/workspace/src
COPY --chown=$USER params.yaml /home/$USER/workspace/params.yaml
COPY --chown=$USER experiments /home/$USER/workspace/experiments
COPY --chown=$USER scripts/entrypoint.sh /home/$USER/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/home/worker/entrypoint.sh"]
