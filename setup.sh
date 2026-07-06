#!/bin/sh

# One-time host setup to run GPU containers under Podman on RHEL-like systems.

echo "Installing prerequisites..."
# Add NVIDIA's RPM repo and install the container toolkit (exposes the GPU to containers).
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo dnf -y install nvidia-container-toolkit

# Generate the CDI spec so Podman can inject the GPU as a device, then list what was found.
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list

# Allow SELinux-confined containers to access host devices (the GPU).
sudo setsebool -P container_use_devices 1
