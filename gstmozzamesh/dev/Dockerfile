# Development image for mozza_mesh: the same base image as the official build
# (../../Dockerfile), with the compiler and headers the builder stage installs,
# plus the prebuilt MediaPipe runtime. Lets the plugin be rebuilt with g++ in
# seconds instead of a full Bazel/MediaPipe build. See gstmozzamesh/dev/build.sh.
FROM ducksouplab/debian-gstreamer:deb12-with-plugins-cuda12.2-gst1.28.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config libglib2.0-dev libopencv-dev python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/*
ENV GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0:/opt/gstreamer/lib/x86_64-linux-gnu/gstreamer-1.0
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/gstreamer/lib/x86_64-linux-gnu
ENV PATH=/opt/gstreamer/bin:$PATH
