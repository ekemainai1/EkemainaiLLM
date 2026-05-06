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

variable "preferred_sizes" {
  description = "Preferred GPU sizes in priority order"
  type        = list(string)
  default = [
    "gpu-mi300x1-192gb",
    "gpu-h100x1-80gb",
    "gpu-h200x1-141gb",
    "gpu-6000adax1-48gb",
    "gpu-l40sx1-48gb",
    "gpu-4000adax1-20gb"
  ]
}

variable "force_region" {
  description = "Force a specific region first (set null to disable force)"
  type        = string
  default     = "atl1"
}

variable "force_size" {
  description = "Force a specific size first (set null to disable force)"
  type        = string
  default     = "gpu-mi300x1-192gb"
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
  api_response = try(jsondecode(data.http.do_sizes.response_body), { sizes : [] })
  all_sizes    = try(local.api_response.sizes, [])

  # GPU sizes discovered from API
  gpu_sizes   = [for s in local.all_sizes : s if strcontains(s.slug, "gpu")]
  gpu_by_slug = { for s in local.gpu_sizes : s.slug => s }

  # First pass: preferred size + preferred region
  preferred_combos = flatten([
    for sz in var.preferred_sizes : [
      for rg in var.preferred_regions : {
        size   = sz
        region = rg
      }
      if contains(keys(local.gpu_by_slug), sz) && contains(local.gpu_by_slug[sz].regions, rg)
    ]
  ])

  # Fallback pass: any discovered GPU in preferred regions
  fallback_combos = flatten([
    for s in local.gpu_sizes : [
      for rg in var.preferred_regions : {
        size   = s.slug
        region = rg
      }
      if contains(s.regions, rg)
    ]
  ])

  # Fallback pass 2: preferred size in any available region
  preferred_size_any_region_combos = [
    for sz in var.preferred_sizes : {
      size   = sz
      region = local.gpu_by_slug[sz].regions[0]
    }
    if contains(keys(local.gpu_by_slug), sz) && length(local.gpu_by_slug[sz].regions) > 0
  ]

  # Fallback pass 3: any GPU in any available region
  any_gpu_any_region_combos = [
    for s in local.gpu_sizes : {
      size   = s.slug
      region = s.regions[0]
    }
    if length(s.regions) > 0
  ]

  # Forced combo (if configured and valid for account/region visibility)
  force_requested = var.force_region != null && var.force_size != null
  force_valid     = local.force_requested && contains(keys(local.gpu_by_slug), var.force_size) && contains(local.gpu_by_slug[var.force_size].regions, var.force_region)
  force_combo = local.force_valid ? {
    size   = var.force_size
    region = var.force_region
  } : null

  selected_combo = local.force_combo != null ? local.force_combo : (
    length(local.preferred_combos) > 0 ? local.preferred_combos[0] : (
      length(local.fallback_combos) > 0 ? local.fallback_combos[0] : (
        length(local.preferred_size_any_region_combos) > 0 ? local.preferred_size_any_region_combos[0] : (
          length(local.any_gpu_any_region_combos) > 0 ? local.any_gpu_any_region_combos[0] : null
        )
      )
    )
  )

  gpu_available   = local.selected_combo != null
  selected_size   = local.gpu_available ? local.selected_combo.size : "none"
  selected_region = local.gpu_available ? local.selected_combo.region : "none"
}

# Single GPU Droplet
resource "digitalocean_droplet" "gpu_training" {
  count   = local.gpu_available ? 1 : 0
  name    = "gpu-training-${formatdate("YYYYMMDD", timestamp())}"
  region  = local.selected_region
  size    = local.selected_size
  image   = "ubuntu-22-04-x64"
  backups = false

  ssh_keys = [var.ssh_key_fingerprint]

  user_data = <<-EOF
#!/bin/bash
set -e

echo "=== GPU Training Node ==="
echo "Region: ${local.selected_region}"
echo "Size: ${local.selected_size}"

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
  value = {
    region                     = local.selected_region
    size                       = local.selected_size
    force_requested            = local.force_requested
    force_valid                = local.force_valid
    discovered_gpu_count       = length(local.gpu_sizes)
    preferred_combo_count      = length(local.preferred_combos)
    preferred_region_combo_cnt = length(local.fallback_combos)
  }
}
