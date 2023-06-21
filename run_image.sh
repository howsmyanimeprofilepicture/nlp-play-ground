docker run --name catchmind-container --rm -dit\
           -p 80:80 \
           --mount type=bind,source="$(pwd)/src",target=/code/src \
           catchmind \
           /bin/bash