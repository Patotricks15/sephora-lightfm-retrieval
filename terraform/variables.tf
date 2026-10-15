variable "project_name" {
  description = "Prefix applied to every resource name."
  type        = string
  default     = "sephora-lightfm"
}

variable "environment" {
  description = "Environment tag (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region. Floci ignores this but the provider requires it."
  type        = string
  default     = "us-east-1"
}

variable "use_floci" {
  description = "Target the local Floci AWS emulator instead of real AWS."
  type        = bool
  default     = true
}

variable "floci_endpoint" {
  description = "Endpoint used for every service when use_floci is true."
  type        = string
  default     = "http://localhost:4566"
}

variable "topn_read_capacity" {
  description = "DynamoDB read capacity units. Ignored when billing is PAY_PER_REQUEST."
  type        = number
  default     = 5

  validation {
    condition     = var.topn_read_capacity > 0
    error_message = "topn_read_capacity must be greater than zero."
  }
}

variable "artifact_retention_days" {
  description = "Days before old model artifact versions are expired from S3."
  type        = number
  default     = 90

  validation {
    condition     = var.artifact_retention_days >= 1
    error_message = "artifact_retention_days must be at least 1."
  }
}

variable "training_schedule" {
  description = "EventBridge cron expression for the nightly retraining job."
  type        = string
  default     = "cron(0 4 * * ? *)"
}
