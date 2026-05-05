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

# Parse and filter for GPU sizes - with error handling
locals {
  api_response   = try(jsondecode(data.http.do_sizes.response_body), {sizes: []})
  all_sizes      = try(local.api_response.sizes, [])
  
  # Filter for GPU sizes - be more inclusive
  gpu_sizes = length(local.all_sizes) > 0 ? [for s in local.all_sizes : s if s.vcpus >= 8 && contains(s.slug, "gpu")] : []
  
  # Find first GPU available (simplified - use first GPU and first preferred region)
  first_gpu = length(local.gpu_sizes) > 0 ? local.gpu_sizes[0] : null
  droplet_config = first_gpu != null ? {
    region = var.preferred_regions[0]
    size   = first_gpu.slug
  } : null
}

# Single GPU Droplet
resource "digitalocean_droplet" "gpu_training" {
  count    = local.droplet_config != null ? 1 : 0
  name     = "gpu-training-${formatdate("YYYYMMDD", timestamp())}"
  region   = local.droplet_config != null ? local.droplet_config.region : "nyc1"
  size     = local.droplet_config != null ? local.droplet_config.size : "s-4vcpu-8gb"
  image    = "ubuntu-22-04-x64"
  backups  = false

  ssh_keys = [var.ssh_key_fingerprint]

  user_data = <<-EOF
#!/bin/bash
set -e

echo "=== GPU Training Node ==="
echo "Region: ${local.droplet_config != null ? local.droplet_config.region : "fallback"}"
echo "Size: ${local.droplet_config != null ? local.droplet_config.size : "fallback"}"

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
  value       = local.droplet_config != null ? local.droplet_config : {region: "nyc1", size: "s-4vcpu-8gb"}
}