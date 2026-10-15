# Serving infrastructure for precomputed hybrid-WARP recommendations.
#
# Design note: scoring is a dot product over a d-dimensional feature-sum
# space, cheap enough to run inside Lambda. We still precompute top-N lists
# into DynamoDB because the ranking also has to exclude each user's review
# history, and doing that at request time would mean loading the full
# interaction set per call. A user/item absent from the precomputed table
# (truly new since the last retrain) is served by the request-time
# feature-based fallback documented in `src/recommender/serving/topn.py`.

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# ---------------------------------------------------------------------------
# Artifacts bucket: model embeddings, feature matrices, evaluation results
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${local.name_prefix}-artifacts"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

// The floci/floci emulator does not converge the propagation check the AWS
// provider polls for here, so a local `apply` hangs for 3 minutes and then
// fails. The rule is real-AWS only; everything else in this stack runs in
// both modes.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  count  = var.use_floci ? 0 : 1
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-model-versions"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.artifact_retention_days
    }
  }
}

# ---------------------------------------------------------------------------
# Top-N table: one item per user, read by the inference Lambda
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "topn" {
  name         = "${local.name_prefix}-topn"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = !var.use_floci
  }

  server_side_encryption {
    enabled = !var.use_floci
  }
}

# ---------------------------------------------------------------------------
# IAM for the inference Lambda
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "inference" {
  name               = "${local.name_prefix}-inference-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

data "aws_iam_policy_document" "inference" {
  # Read-only on the top-N table: the inference path never writes.
  statement {
    effect    = "Allow"
    actions   = ["dynamodb:GetItem", "dynamodb:BatchGetItem"]
    resources = [aws_dynamodb_table.topn.arn]
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/models/*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:*:*:*"]
  }
}

resource "aws_iam_role_policy" "inference" {
  name   = "${local.name_prefix}-inference-policy"
  role   = aws_iam_role.inference.id
  policy = data.aws_iam_policy_document.inference.json
}

# ---------------------------------------------------------------------------
# Inference Lambda
# ---------------------------------------------------------------------------

data "archive_file" "inference" {
  type        = "zip"
  source_file = "${path.module}/../src/recommender/serving/handler.py"
  output_path = "${path.module}/.build/inference.zip"
}

resource "aws_lambda_function" "inference" {
  function_name    = "${local.name_prefix}-inference"
  role             = aws_iam_role.inference.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.12"
  filename         = data.archive_file.inference.output_path
  source_code_hash = data.archive_file.inference.output_base64sha256
  timeout          = 10
  memory_size      = 256

  environment {
    variables = {
      TOPN_TABLE       = aws_dynamodb_table.topn.name
      ARTIFACTS_BUCKET = aws_s3_bucket.artifacts.id
      DEFAULT_K        = "10"
    }
  }
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/aws/lambda/${aws_lambda_function.inference.function_name}"
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# Nightly retraining trigger
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "retrain" {
  name                = "${local.name_prefix}-retrain"
  description         = "Nightly hybrid WARP retraining trigger."
  schedule_expression = var.training_schedule
}
