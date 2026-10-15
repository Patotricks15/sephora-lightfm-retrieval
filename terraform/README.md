# Infrastructure

Terraform for the serving path: an S3 bucket for model artifacts, a DynamoDB
table of precomputed top-N lists, and a Lambda behind IAM that reads them.

The default profile targets the local **Floci** AWS emulator, so `apply`
costs nothing and needs no AWS account.

## Local (default)

```bash
docker compose up -d                       # start Floci on :4566
terraform init
terraform validate
terraform apply -auto-approve
terraform output
terraform destroy -auto-approve
docker compose down
```

## Real AWS

The same configuration deploys to a real account by flipping one variable:

```bash
terraform apply -var use_floci=false -var aws_region=sa-east-1
```

With `use_floci = false` the provider stops injecting dummy credentials
and custom endpoints, and the two features the Floci emulator does not
implement (DynamoDB point-in-time recovery and server-side encryption)
switch on.

## Variables

| Variable | Default | Purpose |
|---|---|---|
| `project_name` | `sephora-lightfm` | Prefix for every resource name |
| `environment` | `dev` | Environment tag and name suffix |
| `aws_region` | `us-east-1` | Target region |
| `use_floci` | `true` | Target the local Floci emulator instead of real AWS |
| `floci_endpoint` | `http://localhost:4566` | Floci emulator address |
| `artifact_retention_days` | `90` | Expiry for non-current model versions in S3 |
| `training_schedule` | `cron(0 4 * * ? *)` | Nightly retraining trigger |

## Design notes

**Why precompute top-N instead of scoring in Lambda for every request?**
Hybrid WARP scoring is a dot product over a d-dimensional feature-sum space
and would run comfortably in Lambda. The ranking, however, must exclude each
user's review history, which would mean loading the full interaction set on
every request. Precomputing moves that cost offline and bounds p99 latency
by DynamoDB rather than by model size.

**What about users/items created after the last retrain?** They have no row
in the top-N table. The Lambda handler returns a 404 with a `fallback` hint
rather than an empty list; the documented recovery path
(`src/recommender/serving/topn.py::score_new_user`) builds that user's
feature vector from profile attributes alone (skin type/tone captured at
signup) and scores the catalogue directly. It is slower than a DynamoDB
GetItem, which is why it is a fallback and not the default path -- the next
scheduled retrain folds the user into the precomputed table.

**Why `PAY_PER_REQUEST`?** Portfolio traffic is bursty and near zero on
average. On-demand billing avoids paying for provisioned capacity that sits
idle, and removes capacity planning from the deployment.

**IAM scope.** The inference role is read-only: `GetItem`/`BatchGetItem` on
the top-N table and `GetObject` restricted to the `models/` prefix. The
request path never writes, so it is not granted write permissions.

## Cost

Everything above is free on Floci. On real AWS at portfolio traffic:
S3 under $0.05/month for less than 1 GB, DynamoDB on-demand under $1/month,
and Lambda within the free tier. `terraform destroy` plus `docker compose
down` removes all of it.
