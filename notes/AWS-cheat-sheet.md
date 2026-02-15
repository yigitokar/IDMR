# AWS Experiment Cheat Sheet

## Project Goal

We are running experiments for the IDMR paper to fill **Table 1**, **Table 2**, **Table 3**, and add an **empirical example** (Yelp data).

---

## Current Status (as of 2026-02-14)

### Table 1: DGP A (High M) — COMPLETE
- **Location:** `s3://idmr-experiments-2025/table1_highM_full/table1/`
- **Config:** n=1000, p=5, M=(500,600), S=10, B=50 seeds, init=pairwise, solver=scs
- **Status:** ✅ All 50 seeds complete for all d values

| d | MSE (mean ± std) |
|---|------------------|
| 250 | 0.0037 ± 0.0018 |
| 500 | 0.0091 ± 0.0041 |
| 1000 | 0.0222 ± 0.0100 |
| 2000 | 0.0533 ± 0.0607 |
| 5000 | 0.0623 ± 0.0123 |

MSE increases monotonically with d (as expected). This fixed the non-monotonicity issue seen with low M.

### Table 1: DGP A (Original Low M) — COMPLETE (reference only)
- **Location:** `s3://idmr-experiments-2025/table1/table1/`
- **Config:** n=1000, p=5, M=(200,300), B=50 seeds

**S=10:**
| d | MSE (mean ± std) |
|---|------------------|
| 250 | 0.0353 ± 0.0035 |
| 500 | 0.0512 ± 0.0118 |
| 1000 | 0.0774 ± 0.0250 |
| 2000 | 0.1139 ± 0.1134 |
| 5000 | 0.0925 ± 0.0085 ← non-monotonic! |

**S=20:**
| d | MSE (mean ± std) |
|---|------------------|
| 250 | 0.0086 ± 0.0014 |
| 500 | 0.0222 ± 0.0084 |
| 1000 | 0.0492 ± 0.0214 |
| 2000 | 0.0644 ± 0.0328 |
| 5000 | 0.0717 ± 0.0043 (38/50 seeds) |

### Table 1: DGP C — COMPLETE
- **Location:** `s3://idmr-experiments-2025/table1/table1/`
- **Config:** n=1000, p=5, B=50 seeds
- **IMPORTANT:** DGP C hardcodes M as mixture of N(10,1) and N(60,5). The `--m-range` flag has NO EFFECT on DGP C. See `idmr_core/simulation.py:97-99`.

**S=10:**
| d | MSE (mean ± std) |
|---|------------------|
| 250 | 453.31 ± 1619.72 |
| 500 | 357.76 ± 1295.62 |
| 1000 | 392.72 ± 1896.86 |
| 2000 | 8.30 ± 13.47 |
| 5000 | 1.35 ± 0.58 |

**S=20:**
| d | MSE (mean ± std) |
|---|------------------|
| 250 | 367.40 ± 1382.80 |
| 500 | 322.82 ± 1178.48 |
| 1000 | 386.98 ± 1865.83 |
| 2000 | 8.38 ± 14.81 |
| 5000 | 1.48 ± 0.71 (48/50 seeds) |

### Table 2 — NOT STARTED
- **Status:** ⏳ Requires GPU instances
- **What:** SGD-based estimator experiments

### Table 3 — PARTIALLY COMPLETE
- **Location:** `s3://idmr-experiments-2025/table3/table3/`
- **Config:** d=[200,250,500,1000,2000], p=[50,100,500,1000,2000], lambda=[0,0.01,0.1], S=10, B=50
- **Status:** Only d=200 completed (14/75 configs). Runs were killed when machines were repurposed.
- **Completed configs (all d=200):**

| p | λ=0 | λ=0.01 | λ=0.1 |
|---|-----|--------|-------|
| 50 | 50/50 ✅ | 50/50 ✅ | 50/50 ✅ |
| 100 | 50/50 ✅ | 50/50 ✅ | 50/50 ✅ |
| 500 | 50/50 ✅ | 50/50 ✅ | 50/50 ✅ |
| 1000 | 50/50 ✅ | 22/50 ⚠️ | 50/50 ✅ |
| 2000 | 42/50 ⚠️ | missing | 37/50 ⚠️ |

- **TODO:** Rerun d=250,500,1000,2000 (61 remaining configs). Also need to complete partial d=200 configs.

### Empirical Example — NOT STARTED
- **Status:** ⏳ Yelp data loading code exists (`scripts/load_yelp.py`)

---

## AWS Infrastructure

### Region
```
us-east-2 (Ohio)
```

### Running Instances (as of 2026-02-14)
All instances are IDLE (no experiments running). 8 total.

| IP | Type | vCPUs | Instance ID | Name | Disk | Notes |
|----|------|-------|-------------|------|------|-------|
| 52.14.191.59 | c6i.8xlarge | 32 | i-0c2b62667697ec31a | idmr-rerun-d5000 | 8GB | Original, has IDMR + AWS CLI |
| 3.16.218.20 | c6i.8xlarge | 32 | i-0c2fe24d70b33aaa3 | idmr-table3 | 8GB | Has IDMR + AWS CLI |
| 18.190.176.83 | c6i.8xlarge | 32 | i-0299ac777be7767e9 | idmr-table3 | 8GB | Has IDMR + AWS CLI |
| 3.135.196.71 | c6i.8xlarge | 32 | i-02586ef5dd80f4ad8 | idmr-table3 | 8GB | Has IDMR + AWS CLI |
| 3.21.237.180 | c6i.8xlarge | 32 | i-012ee8cc62aa63547 | idmr-dgpc-highM | 50GB | Has IDMR + AWS CLI |
| 3.135.244.241 | c6i.8xlarge | 32 | i-0515029416349fc86 | idmr-dgpc-highM | 50GB | Has IDMR + AWS CLI |
| 18.188.126.76 | c6i.8xlarge | 32 | i-09a63463ea1d244b4 | idmr-dgpc-highM | 50GB | Has IDMR + AWS CLI |
| 3.133.102.212 | c6i.8xlarge | 32 | i-0d49669f0bc5fcc42 | idmr-dgpc-highM | 50GB | Has IDMR + AWS CLI |

### vCPU Quota
- **On-Demand Standard:** 256 vCPUs max
- **Currently using:** 256 vCPUs (8 × 32) — AT LIMIT
- **To launch more:** Must terminate some first

### Cost
- c6i.8xlarge: ~$1.36/hour each
- 8 instances: ~$10.88/hour (~$261/day)
- **IMPORTANT:** Terminate idle instances to save money!

---

## Credentials & Access

### SSH Key
```
~/.ssh/idmr-key.pem
```

### SSH to Instance
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP_ADDRESS>
```

### AWS CLI (local)
AWS credentials are configured in the default profile. No additional setup needed.

### AWS CLI (on instances)
- The OLD instances (52.14.191.59, 3.16.218.20, 18.190.176.83, 3.135.196.71) have AWS credentials configured via instance metadata or profile.
- The NEW instances (3.21.237.180, 3.135.244.241, 18.188.126.76, 3.133.102.212) have AWS CLI installed but **NO credentials configured**. S3 uploads will fail! You must either:
  1. Configure credentials: `aws configure` on each instance
  2. Or use `--local-output-dir` and scp results back

---

## Common Commands

### Check Running Instances
```bash
aws ec2 describe-instances --region us-east-2 --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[PublicIpAddress,InstanceType,InstanceId,Tags[?Key==`Name`].Value|[0]]' --output table
```

### List S3 Results
```bash
# Table 1 DGP A (high M) — FINAL RESULTS
aws s3 ls s3://idmr-experiments-2025/table1_highM_full/table1/ --region us-east-2

# Table 1 (original M, both DGP A and C)
aws s3 ls s3://idmr-experiments-2025/table1/table1/ --region us-east-2

# Table 3
aws s3 ls s3://idmr-experiments-2025/table3/table3/ --region us-east-2
```

### Check MSE for a File
```bash
aws s3 cp s3://idmr-experiments-2025/table1_highM_full/table1/dgp_A_d_500_S_10_init_pairwise_highM.csv - --region us-east-2 \
  | cut -d',' -f28 | tail -n+2 \
  | awk '{s+=$1; ss+=$1*$1; n++} END {printf "n=%d mean=%.4f std=%.4f\n", n, s/n, sqrt(ss/n - (s/n)^2)}'
```

### Check All Table 1 DGP A Results at Once
```bash
for d in 250 500 1000 2000 5000; do
  aws s3 cp s3://idmr-experiments-2025/table1_highM_full/table1/dgp_A_d_${d}_S_10_init_pairwise_highM.csv - --region us-east-2 2>/dev/null \
    | cut -d',' -f28 | tail -n+2 \
    | awk -v d=$d '{s+=$1; ss+=$1*$1; n++} END {printf "d=%d: n=%d mean=%.4f std=%.4f\n", d, n, s/n, sqrt(ss/n - (s/n)^2)}'
done
```

### Check Experiment Progress on Instance
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP> "ps aux | grep run_experiments | grep -v grep"
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP> "tail -5 /tmp/run1.log"
```

### Start an Experiment (use nohup, not tmux — tmux had issues)
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP> 'cd ~/IDMR && nohup bash -c "export PATH=\$HOME/.local/bin:\$PATH && uv run python scripts/run_experiments.py --table 1 --dgp A --d 500 --S 10 --B 50 --m-range 500 600 --n-workers 30 --output-dir s3://idmr-experiments-2025/table1_highM_full/ --output-suffix highM" > /tmp/run1.log 2>&1 &'
```

### Kill Experiment
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP> "pkill -9 -f run_experiments"
```

### Terminate Instance
```bash
aws ec2 terminate-instances --region us-east-2 --instance-ids <INSTANCE_ID>
```

### Launch New Instance (with 50GB disk)
```bash
aws ec2 run-instances \
  --region us-east-2 \
  --image-id ami-0ea3c35c5c3284d82 \
  --instance-type c6i.8xlarge \
  --key-name idmr-key \
  --security-group-ids sg-012b1ca1a143d9eea \
  --subnet-id subnet-057d910c8731eb2ab \
  --count 1 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=idmr-experiment}]' \
  --query 'Instances[*].[InstanceId,PublicIpAddress]' --output text
```

### Setup New Instance (after launch, wait ~30s for boot)
```bash
IP=<new_ip>
ssh -i ~/.ssh/idmr-key.pem -o StrictHostKeyChecking=no ubuntu@$IP "
  curl -LsSf https://astral.sh/uv/install.sh | sh &&
  git clone https://github.com/yigitokar/IDMR.git &&
  cd IDMR && ~/.local/bin/uv sync &&
  sudo apt-get update -qq && sudo apt-get install -qq -y unzip &&
  curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o /tmp/awscliv2.zip &&
  cd /tmp && unzip -q awscliv2.zip && sudo ./aws/install
"
# Then configure AWS credentials:
ssh -i ~/.ssh/idmr-key.pem ubuntu@$IP "aws configure"
```

---

## S3 Bucket Structure

```
s3://idmr-experiments-2025/
├── table1/                        # Original Table 1 runs (low M=200-300)
│   └── table1/
│       ├── dgp_A_d_{250..5000}_S_{10,20}_init_pairwise.csv
│       └── dgp_C_d_{250..5000}_S_{10,20}_init_pairwise.csv
├── table1_highM_full/             # ★ Table 1 DGP A FINAL (M=500-600)
│   └── table1/
│       └── dgp_A_d_{250..5000}_S_10_init_pairwise_highM.csv
├── table1_highM/                  # Early test run (10 seeds, ignore)
├── table1_rerun/                  # Debug rerun (ignore)
├── table1_rerun2/                 # Debug rerun with different seed (ignore)
└── table3/                        # Table 3 runs (only d=200 complete)
    └── table3/
        └── dgp_A_d_200_p_{50..2000}_lam_{0,0p01,0p1}_S_10.csv
```

**CSV columns (28 = MSE):** table, dgp, rep, seed, theta_seed, n, d, p, m_min, m_max, method, solver, init, S, S_effective, optimizer, lr, lambda, epochs, batch_size, device, n_workers, time_total, time_per_iter, time_per_epoch, init_time, final_loss, **mse**, status, error, timestamp

---

## Experiment Script Options

```bash
uv run python scripts/run_experiments.py \
    --table 1                    # Table number (1, 2, or 3)
    --dgp A                      # DGP type (A or C)
    --d 500 1000 2000            # Dimension values
    --S 10                       # Number of slices
    --B 50                       # Number of seeds/repetitions
    --m-range 500 600            # M range (ONLY works for DGP A, ignored by DGP C)
    --n-workers 30               # Parallel workers
    --base-seed 1000             # Starting seed (seeds = base_seed, base_seed+1, ...)
    --output-dir s3://...        # S3 output (needs credentials)
    --local-output-dir /tmp/out  # Local output (always works)
    --output-suffix highM        # Suffix appended to filename
    --partition 0                # For splitting across machines (0-indexed)
    --n-partitions 3             # Total number of partitions
    --solver scs                 # CVXPY solver (scs, mosek, clarabel)
```

---

## Known Issues & Lessons Learned

1. **M parameter matters for DGP A:** Low M (200-300) caused non-monotonic MSE at d=5000. Use M=(500,600) for clean results. The `--m-range` flag controls this.

2. **DGP C ignores `--m-range`:** DGP C generates M from a hardcoded mixture distribution (N(10,1) and N(60,5)). To change M for DGP C, you must modify `idmr_core/simulation.py:97-99`.

3. **AWS credentials on new instances:** New instances launched after the original 4 do NOT have AWS credentials. S3 uploads will fail with "Unable to locate credentials". Must run `aws configure` after setup.

4. **tmux unreliable for background jobs:** tmux sessions sometimes die silently when launched via SSH. Use `nohup ... &` instead, and check `/tmp/run*.log` for output.

5. **Disk space:** Default AMI has ~7GB root disk. torch with CUDA needs ~5GB. Launch instances with `--block-device-mappings` to get 50GB.

6. **Instance IPs change on stop/start:** If you stop and restart instances, public IPs change. Use instance IDs for reliable identification.

7. **Seeds:** Default base seed is 1000. All runs used seeds 1000-1049 for B=50.

8. **Cost awareness:** 8 × c6i.8xlarge = ~$261/day. Terminate when not in use.

---

## TODO (Priority Order)

1. **Configure AWS credentials on new instances** (or terminate them if not needed)
2. **Rerun DGP C with higher M** — requires modifying `idmr_core/simulation.py` to accept M_range for DGP C
3. **Complete Table 3** — d=250,500,1000,2000 still needed (61 configs), plus partial d=200 configs
4. **Table 2** — requires GPU instances (SGD estimator)
5. **Empirical example** — Yelp data, loading code exists
6. **Terminate idle instances** to save cost
