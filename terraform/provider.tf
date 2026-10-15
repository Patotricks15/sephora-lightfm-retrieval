# Provider wiring.
#
# The default profile targets the local Floci AWS emulator on localhost:4566
# with dummy credentials, so `terraform apply` costs nothing and needs no AWS
# account. Set `use_floci = false` to point the exact same resources at real
# AWS.

provider "aws" {
  region = var.aws_region

  access_key = var.use_floci ? "test" : null
  secret_key = var.use_floci ? "test" : null

  # Floci does not implement credential/account validation.
  skip_credentials_validation = var.use_floci
  skip_metadata_api_check     = var.use_floci
  skip_requesting_account_id  = var.use_floci
  s3_use_path_style           = var.use_floci

  dynamic "endpoints" {
    for_each = var.use_floci ? [1] : []
    content {
      s3         = var.floci_endpoint
      dynamodb   = var.floci_endpoint
      lambda     = var.floci_endpoint
      iam        = var.floci_endpoint
      sts        = var.floci_endpoint
      logs       = var.floci_endpoint
      events     = var.floci_endpoint
      apigateway = var.floci_endpoint
    }
  }

  default_tags {
    tags = {
      Project     = var.project_name
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }
}
