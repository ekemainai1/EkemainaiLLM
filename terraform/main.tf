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

# Query available GPU sizes from DigitalOcean API
data "external" "available_gpu" {
  program = ["bash", "-lc", <<EOT
curl -s -H "Authorization: Bearer $DO_TOKEN" \
  "https://api.digitalocean.com/v2/sizes?per_page=200" \
  | jq -r '.sizes[] | select(.memory >= 190000) | "\(.slug) \(.regions | join(","))"'
EOT
  ]
  env = {
    DO_TOKEN = var.do_token
  }
}

# Find first available GPU in preferred regions
locals {
  available_gpus = { for line in split("\n", data.external.available_gpu.result) : 
    split(" ", line)[0] => split(",", split(" ", line)[1]) 
    if length(split(" ", line)) > 1
  }
  
  # Find GPU that exists in preferred regions
  selected = {
    for region in var.preferred_regions :
    region => [for size in keys(local.available_gpus) : size if contains(local.available_gpus[size], region)]
    if length([for size in keys(local.available_gpus) : size if contains(local.available_gpus[size], region)]) > 0
  }
  
  selected_region = keys(local.selected)[0]
  selected_size  = local.selected[local.selected_region][0]
  
  droplet_config = {
    region = local.selected_region
    size   = local.selected_size
  }
}

# Single GPU Droplet
resource "digitalocean_droplet" "gpu_training" {
  name     = "gpu-training-${formatdate("YYYYMMDD", timestamp())}"
  region   = local.droplet_config.region
  size     = local.droplet_config.size
  image    = "ubuntu-22-04-x64"
  backups  = false

  ssh_keys = [var.ssh_key_fingerprint]

  user_data = <<-EOF
#!/bin/bash
set -e

echo "=== GPU Training Node ==="
echo "Region: ${local.droplet_config.region}"
echo "Size: ${local.droplet_config.size}"

# Update and install basics
apt-get update
apt-get install -y git python3-pip curl wget tmux htop

# Install PyTorch with ROCm support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1 || true

# Install training dependencies
pip3 install transformers accelerate peft datasets sentencepiece || true

# Clone repository
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
  value       = digitalocean_droplet.gpu_training.ipv4_address
}

output "gpu_info" {
  description = "GPU configuration"
  value       = {
    region = local.droplet_config.region
    size   = local.droplet_config.size
  }
}