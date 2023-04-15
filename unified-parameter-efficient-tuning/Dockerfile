FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6
WORKDIR /

# Copies all files in current cwd where docker file lies into folder param-efficient-fine-tuning in the docker image.
COPY . /param-efficient-fine-tuning

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
WORKDIR ./param-efficient-fine-tuning
RUN pip install -e .
RUN pip install datasets==1.11.0

# Sets up the entry point to invoke the trainer.
#ENTRYPOINT ["bin/bash", "/exps/run_glue.sh"]
#CMD ["/bin/bash"]
#RUN ["chmod", "+x", "/exps/run_glue.sh"]
#RUN ./exps/run_glue.sh

ENTRYPOINT ["sh", "./exps/run_glue.sh"]
#RUN chomd +x script.sh
#RUN chomd +x /exps/run_glue.sh
#terminal
#docker ps
#docker exec -it <container name> /bin/bash
#docker logs [OPTIONS]