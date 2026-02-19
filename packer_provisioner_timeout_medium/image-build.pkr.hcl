packer {
  required_plugins {
    docker = {
      version = ">= 1.0.8"
      source  = "github.com/hashicorp/docker"
    }
  }
}

source "docker" "ubuntu" {
  image  = "ubuntu:20.04"
  commit = true
}

build {
  sources = ["source.docker.ubuntu"]

  provisioner "shell" {
    inline = [
      "echo 'Starting provisioning process'",
      "ls -la /tmp"
    ]
    timeout = "60s"
  }

  provisioner "shell" {
    inline = [
      "apt-get update",
      "apt-get install curl"
    ]
    timeout = "60s"
  }

  provisioner "shell" {
    inline = [
      "echo 'Installation complete'",
      "curl --version"
    ]
    timeout = "60s"
  }
}