output "artifacts_bucket" {
  description = "S3 bucket holding model embeddings, feature matrices and results."
  value       = aws_s3_bucket.artifacts.id
}

output "topn_table_name" {
  description = "DynamoDB table serving precomputed top-N lists."
  value       = aws_dynamodb_table.topn.name
}

output "inference_function_name" {
  description = "Lambda function name for the recommendation endpoint."
  value       = aws_lambda_function.inference.function_name
}

output "inference_role_arn" {
  description = "Execution role used by the inference Lambda."
  value       = aws_iam_role.inference.arn
}

output "endpoint_mode" {
  description = "Whether the stack is pointed at Floci or real AWS."
  value       = var.use_floci ? "floci (${var.floci_endpoint})" : "aws (${var.aws_region})"
}
