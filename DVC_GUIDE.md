# DVC Configuration Guide

## What is DVC?

DVC (Data Version Control) is like Git for data and models. It helps you:
- Version control large files (datasets, models)
- Track experiments and reproduce results
- Share data efficiently with your team
- Create reproducible ML pipelines

## ✅ DVC Setup Complete

### What We've Configured:

1. **DVC Initialized** ✅
   - `.dvc` directory created
   - DVC tracking enabled

2. **Pipeline Defined** (`dvc.yaml`) ✅
   ```yaml
   stages:
     - data_ingestion
     - data_transformation
     - model_training
   ```

3. **Data Tracked** ✅
   - `data/raw/ecommerce_churn.csv.dvc`
   - `models/best_model.pkl.dvc`

## 📁 DVC Files Created

- `.dvc/` - DVC configuration directory
- `.dvcignore` - Files to ignore (like .gitignore)
- `dvc.yaml` - Pipeline definition
- `dvc.lock` - Pipeline lock file (auto-generated)
- `*.dvc` files - Pointers to tracked files

## 🚀 Using DVC

### Run the Complete Pipeline

```bash
dvc repro
```

This will:
1. Download data (if needed)
2. Run data transformation
3. Train models
4. Track all outputs

### Track New Data/Models

```bash
# Track a file
dvc add data/new_dataset.csv

# Commit the .dvc file to git
git add data/new_dataset.csv.dvc .gitignore
git commit -m "Track new dataset"
```

### Setup Remote Storage (Optional)

#### Option 1: Local Remote (for testing)
```bash
dvc remote add -d myremote /path/to/storage
```

#### Option 2: Google Drive
```bash
dvc remote add -d myremote gdrive://folder_id
dvc remote modify myremote gdrive_use_service_account true
```

#### Option 3: AWS S3
```bash
dvc remote add -d myremote s3://mybucket/path
```

#### Option 4: DagsHub (Recommended for ML)
```bash
# Create account at dagshub.com
# Create repository
dvc remote add -d origin https://dagshub.com/username/repo.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user your_username
dvc remote modify origin --local password your_token
```

### Push Data to Remote

```bash
dvc push
```

### Pull Data from Remote

```bash
dvc pull
```

## 🔄 Typical Workflow

```bash
# 1. Make changes to code
vim src/components/model_trainer.py

# 2. Run pipeline
dvc repro

# 3. Track changes
git add dvc.lock
git commit -m "Improved model accuracy"

# 4. Push data and code
dvc push
git push
```

## 📊 View Pipeline

```bash
# Show pipeline DAG
dvc dag

# Show pipeline status
dvc status

# Show metrics
dvc metrics show
```

## 🎯 Benefits for This Project

1. **Data Versioning**: Track different versions of `ecommerce_churn.csv`
2. **Model Versioning**: Track different trained models
3. **Reproducibility**: Anyone can reproduce your exact results
4. **Collaboration**: Share large files without bloating Git repo
5. **Experimentation**: Try different features, compare results

## 🔧 Current Pipeline Stages

### Stage 1: Data Ingestion
- **Command**: `python -m src.ecommerce_customer_churn.components.data_ingestion`
- **Output**: `data/raw/ecommerce_churn.csv`

### Stage 2: Data Transformation
- **Command**: `python -m src.ecommerce_customer_churn.components.data_transformation`
- **Dependencies**: Raw data, transformation script
- **Outputs**: Processed features (Phase 1, Phase 2), imputation values, encoders

### Stage 3: Model Training
- **Command**: `python -m src.ecommerce_customer_churn.components.model_trainer`
- **Dependencies**: Processed data, training script
- **Outputs**: Best model, results JSON

## 📝 Next Steps

1. **Setup Remote Storage** (recommended: DagsHub)
   ```bash
   dvc remote add -d origin <your-remote-url>
   ```

2. **Push Data**
   ```bash
   dvc push
   ```

3. **Commit DVC Files**
   ```bash
   git add .dvc .dvcignore dvc.yaml *.dvc
   git commit -m "Configure DVC for data versioning"
   git push
   ```

## 🆘 Common Commands

```bash
# Check DVC status
dvc status

# Run pipeline
dvc repro

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff

# List remotes
dvc remote list

# Check out specific version
git checkout <commit>
dvc checkout
```

## 🎓 Learn More

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DagsHub](https://dagshub.com/) - Free DVC remote storage for ML

---

**Status**: ✅ DVC Configured and Ready to Use!
