# AWS Experiment Cheat Sheet

## Project Goal

We are running experiments for the IDMR paper to fill **Table 1**, **Table 2**, **Table 3**, and add an **empirical example**.

---

## Current Status

### Table 1: DGP A (High M) - COMPLETE
- **Location:** `s3://idmr-experiments-2025/table1_highM_full/table1/`
- **Config:** M=(500,600), S=10, B=50 seeds
- **Status:** ✅ Complete

| d | MSE (mean ± std) |
|---|------------------|
| 250 | 0.0037 ± 0.0018 |
| 500 | 0.0091 ± 0.0041 |
| 1000 | 0.0222 ± 0.0100 |
| 2000 | 0.0533 ± 0.0607 |
| 5000 | 0.0626 ± 0.0116 |

### Table 1: DGP C - COMPLETE
- **Location:** `s3://idmr-experiments-2025/table1/table1/`
- **Config:** M=(200,300), S=10 and S=20, B=50 seeds
- **Status:** ✅ Complete

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
| 5000 | 1.48 ± 0.71 |

### Table 2 - PENDING
- **Status:** ⏳ Not started (requires GPU)
- **Note:** Need to set up GPU instances for this

### Table 3 - IN PROGRESS
- **Location:** `s3://idmr-experiments-2025/table3/table3/`
- **Config:** d=[200,250,500,1000,2000], p=[50,100,500,1000,2000], lambda=[0,0.01,0.1]
- **Status:** 🔄 Partially complete

### Empirical Example - PENDING
- **Status:** ⏳ Not started

---

## AWS Infrastructure

### Region
```
us-east-2 (Ohio)
```

### Running Instances
| IP | Type | vCPUs | Instance ID |
|----|------|-------|-------------|
| 52.14.191.59 | c6i.8xlarge | 32 | i-0c2b62667697ec31a |
| 3.16.218.20 | c6i.8xlarge | 32 | i-0c2fe24d70b33aaa3 |
| 18.190.176.83 | c6i.8xlarge | 32 | i-0299ac777be7767e9 |
| 3.135.196.71 | c6i.8xlarge | 32 | i-02586ef5dd80f4ad8 |

### vCPU Quota
- **On-Demand Standard:** 256 vCPUs
- **Currently using:** 128 vCPUs (4 × 32)
- **Available:** 128 vCPUs

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

### AWS CLI
AWS credentials are configured in the default profile. No additional setup needed.

---

## Common Commands

### Check Running Instances
```bash
aws ec2 describe-instances --region us-east-2 --filters "Name=instance-state-name,Values=running" --query 'Reservations[].Instances[].[PublicIpAddress,InstanceType,InstanceId]' --output table
```

### List S3 Results
```bash
# Table 1 DGP A (high M)
aws s3 ls s3://idmr-experiments-2025/table1_highM_full/table1/ --region us-east-2

# Table 1 DGP C
aws s3 ls s3://idmr-experiments-2025/table1/table1/ --region us-east-2

# Table 3
aws s3 ls s3://idmr-experiments-2025/table3/table3/ --region us-east-2
```

### Download Results
```bash
aws s3 cp s3://idmr-experiments-2025/table1_highM_full/table1/dgp_A_d_500_S_10_init_pairwise_highM.csv . --region us-east-2
```

### Check MSE for a File
```bash
aws s3 cp s3://idmr-experiments-2025/table1_highM_full/table1/dgp_A_d_500_S_10_init_pairwise_highM.csv - --region us-east-2 | cut -d',' -f28 | tail -n+2 | awk '{s+=$1; n++} END {printf "Mean MSE: %.4f\n", s/n}'
```

### Check Experiment Progress on Instance
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@52.14.191.59 "ps aux | grep run_experiments | grep -v grep"
ssh -i ~/.ssh/idmr-key.pem ubuntu@52.14.191.59 "tmux capture-pane -t run -p | tail -10"
```

### Start an Experiment
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP> "cd ~/IDMR && tmux new-session -d -s run 'export PATH=\"\$HOME/.local/bin:\$PATH\" && uv run python scripts/run_experiments.py --table 1 --dgp A --d 500 --S 10 --B 50 --m-range 500 600 --n-workers 30 --output-dir s3://idmr-experiments-2025/table1_highM_full/ --output-suffix highM'"
```

### Kill Experiment
```bash
ssh -i ~/.ssh/idmr-key.pem ubuntu@<IP> "pkill -9 -f run_experiments"
```

---

## S3 Bucket Structure

```
s3://idmr-experiments-2025/
├── table1/                    # Original Table 1 runs (low M)
│   └── table1/
│       ├── dgp_A_d_*_S_*_init_pairwise.csv
│       └── dgp_C_d_*_S_*_init_pairwise.csv
├── table1_highM_full/         # Table 1 DGP A with M=(500,600)
│   └── table1/
│       └── dgp_A_d_*_S_10_init_pairwise_highM.csv
├── table3/                    # Table 3 runs
│   └── table3/
│       └── dgp_A_d_*_p_*_lam_*_S_10.csv
└── table1_rerun*/             # Debug/test runs (can ignore)
```

---

## Experiment Script Options

```bash
uv run python scripts/run_experiments.py \
    --table 1                    # Table number (1, 2, or 3)
    --dgp A                      # DGP type (A, B, C)
    --d 500 1000 2000            # Dimension values
    --S 10                       # Number of slices
    --B 50                       # Number of seeds/repetitions
    --m-range 500 600            # M range (observations per doc)
    --n-workers 30               # Parallel workers
    --base-seed 1000             # Starting seed
    --output-dir s3://...        # Output location
    --output-suffix highM        # Suffix for output files
    --partition 0                # For splitting across machines
    --n-partitions 3             # Total number of partitions
```

---

## Important Notes

1. **M parameter matters:** Low M (200-300) caused non-monotonic MSE behavior. Use M=(500,600) for DGP A.

2. **Seeds:** Default seeds start at 1000. Use `--base-seed` to change.

3. **Partitioning:** For large experiments, use `--partition` and `--n-partitions` to split across machines.

4. **tmux sessions:** Experiments run in tmux. Use `tmux attach -t run` to view, `Ctrl-B D` to detach.

5. **Instance cost:** c6i.8xlarge costs ~$1.36/hour. 4 instances = ~$5.44/hour.
