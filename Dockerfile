FROM ubuntu:20.04 AS venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3.8-venv

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install torchvision && pip install -r requirements.txt

FROM ubuntu:20.04 AS model

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates

RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P /pretrained_models

FROM ubuntu:20.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip  libgl1 libglib2.0-0 python3-distutils && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/

COPY --from=venv /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app/
# RUN mkdir -p results/img results/faces
COPY --from=model /pretrained_models experiments/pretrained_models
COPY . .

RUN /opt/venv/bin/python3 setup.py develop
RUN /opt/venv/bin/python3 inference_gfpgan.py --upscale 2 --test_path inputs/one_img --save_root results

ENTRYPOINT [ "/opt/venv/bin/python3" ]
CMD [ "server.py" ]
