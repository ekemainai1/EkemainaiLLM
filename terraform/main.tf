# Terraform configuration for AMD MI300X GPU Droplet
# DigitalOcean GPU Droplet with AMD Instinct MI300X

terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "ssh_key_path" {
  description = "Path to SSH public key"
  type        = string
  default    = "~/.ssh/id_rsa.pub"
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default    = "atl1"  # NYC3, SGP1, AMS3, BLR1, ATL1, etc.
}

# Single MI300X GPU Droplet (192GB VRAM)
resource "digitalocean_droplet" "mi300x_single" {
  name     = "mi300x-training-${formatdate("YYYYMMDD", timestamp())}"
  region  = var.region
  image   = "gpu-amd-base"  # ROCm AI/ML base image
  size    = "gpu-mi300x1-192gb"
  backups = false

  ssh_keys = [var.ssh_key_fingerprint]

  user_data = <<-EOF
              #!/bin/bash
              set -e
              
              echo "=== AMD MI300X GPU Training Node ==="
              
              # Update and install basics
              apt-get update
              apt-get install -y git python3-pip curl wget tmux htop
              
              # Verify ROCm
              echo "Checking ROCm..."
              rocm-smi || echo "ROCm not in PATH"
              
              # Verify GPU
              echo "Checking GPU..."
              rocminfo || echo "rocminfo not available"
              
              # Install PyTorch with ROCm
              echo "Installing PyTorch with ROCm support..."
              pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
              
              # Install training dependencies
              echo "Installing training dependencies..."
              pip3 install transformers accelerate peft datasets sentencepiece
              
              # Clone repository
              echo "Cloning EkemainaiAgent..."
              cd /root
              git clone https://github.com/YOUR_USERNAME/EkemainaiAgent.git
              cd EkemainaiAgent
              
              # Create data symlink (if using volume)
              # ln -s /mnt/data data
              
              echo "=== Setup Complete ==="
              echo "GPU info:"
              rocm-smi
              EOF

  tags = ["ai-training", "gpu", "mi300x"]

  lifecycle {
    create_before_destroy = true
  }
}

# 8x MI300X GPU Droplet for large-scale training
resource "digitalocean_droplet" "mi300x_cluster" {
  count    = var.enable_cluster ? 1 : 0
  name     = "mi300x-cluster-${formatdate("YYYYMMDD", timestamp())}"
  region  = var.region
  image   = "gpu-amd-base"
  size    = "gpu-mi300x8-1536gb"
  backups = false

  ssh_keys = [var.ssh_key_fingerprint]

  user_data = <<-EOF
              #!/bin/bash
              set -e
              
              echo "=== AMD MI300X 8-GPU Training Node ==="
              
              # Update and install basics
              apt-get update
              apt-get install -y git python3-pip curl wget tmux htop
              
              # Verify ROCm
              echo "Checking ROCm..."
              rocm-smi
              
              # Install PyTorch with ROCm
              pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
              
              # Install training dependencies
              pip3 install transformers accelerate peft datasets sentencepiece vllm
              
              # Enable IPv4 forwarding for distributed training
              echo 1 > /proc/sys/net/ipv4/ip_forward
              
              echo "=== Setup Complete ==="
              echo "GPU info:"
              rocm-smi
              EOF

  tags = ["ai-training", "gpu", "mi300x-cluster"]

  lifecycle {
    create_before_destroy = true
  }
}

variable "enable_cluster" {
  description = "Enable 8-GPU cluster node"
  type        = bool
  default    = false
}

variable "ssh_key_fingerprint" {
  description = "DigitalOcean SSH key fingerprint"
  type        = string
  # Get from: doctl compute ssh-key list
}

# Optional: Attached volume for datasets
resource "digitalocean_volume" "training_data" {
  count      = var.enable_volume ? 1 : 0
  name       = "training-data-${formatdate("YYYYMMDD", timestamp())}"
  region     = var.region
  size       = 100
  description = "Training data volume"
  
  tags = ["training-data"]
}

variable "enable_volume" {
  description = "Enable data volume"
  type        = bool
  default    = false
}

# Outputs
output "droplet_ip" {
  description = "Primary GPU Droplet IP"
  value       = digitalocean_droplet.mi300x_single.ipv4_address
}

output "cluster_ips" {
  description = "Cluster node IPs"
  value       = digitalocean_droplet.mi300x_cluster[*].ipv4_address
}

output "gpu_info" {
  description = "GPU configuration"
  value       = {
    slug       = digitalocean_droplet.mi300x_single.size
    vcpus      = digitalocean_droplet.mi300x_single.vcpus
    memory_gb = digitalocean_droplet.mi300x_single.memory / 1024
  }
}

# Usage example in comments:
# 
# # Apply with:
# terraform init
# terraform plan -var="do_token=$DO_TOKEN" -var="ssh_key_fingerprint=xx:xx:xx:xx:xx"
# terraform apply -var="do_token=$DO_TOKEN" -var="ssh_key_fingerprint=xx:xx:xx:xx:xx"
# 
# # Connect via SSH:
# ssh root@<droplet_ip>
# 
# # Run training:
# cd /root/EkemainaiAgent
# python3 scripts/train.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --dataset data/combined_final.jsonl \
#   --output ./fine-tuned-model \
#   --epochs 3 \
#   --batch_size 4