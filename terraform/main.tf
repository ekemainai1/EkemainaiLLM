# Terraform configuration for GPU Droplet
# Dynamically discovers available GPU sizes

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

variable "ssh_key_fingerprint" {
  description = "DigitalOcean SSH key fingerprint"
  type        = string
}

variable "preferred_regions" {
  description = "Preferred regions for GPU droplet"
  type        = list(string)
  default     = ["atl1", "nyc1", "sfo2", "ams3", "sgp1"]
}

# Query all available sizes from DigitalOcean API
data "http" "do_sizes" {
  url = "https://api.digitalocean.com/v2/sizes?per_page=200"
  request_headers = {
    Authorization = "Bearer ${var.do_token}"
  }
}

# Parse and filter for GPU sizes
locals {
  all_sizes  = jsondecode(data.http.do_sizes.body).sizes
  gpu_sizes  = [for s in local.all_sizes : s if s.memory >= 190000]
  
  # Find first GPU available in preferred regions
  droplet_config = length(local.gpu_sizes) > 0 ? {
    region = [for r in var.preferred_regions : r if contains(local.gpu_sizes[0].regions, r)][0]
    size   = local.gpu_sizes[0].slug
  } : null
  
  # Fallback if no GPU found - use default
  final_config = local.droplet_config != null ? local.droplet_config : {
    region = "nyc1"
    size   = "s-4vcpu-8gb"
  }
}

# Single GPU Droplet
resource "digitalocean_droplet" "gpu_training" {
  count    = local.droplet_config != null ? 1 : 0
  name     = "gpu-training-${formatdate("YYYYMMDD", timestamp())}"
  region   = local.final_config.region
  size     = local.final_config.size
  image    = "ubuntu-22-04-x64"
  backups  = false

  ssh_keys = [var.ssh_key_fingerprint]

  user_data = <<-EOF
#!/bin/bash
set -e

echo "=== GPU Training Node ==="
echo "Region: ${local.final_config.region}"
echo "Size: ${local.final_config.size}"

# Update and install basics
apt-get update
apt-get install -y git python3-pip curl wget tmux htop

# Install PyTorch with ROCm support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1 || true

# Install training dependencies
pip3 install transformers accelerate peft datasets sentencepiece || true

echo "=== Setup Complete ==="
EOF

  tags = ["ai-training", "gpu", "training-node"]

  lifecycle {
    create_before_destroy = true
  }
}

# Outputs
output "droplet_ip" {
  description = "GPU Droplet IP"
  value       = length(digitalocean_droplet.gpu_training) > 0 ? digitalocean_droplet.gpu_training[0].ipv4_address : "none"
}

output "gpu_info" {
  description = "GPU configuration"
  value       = local.final_config
}