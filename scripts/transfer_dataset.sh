#!/bin/bash
# scripts/transfer_dataset.sh - Transfer and prepare dataset on MSI
# Run this script from your local machine!

# Configuration - REPLACE THESE VALUES
MSI_USERNAME="yourusername"  # Replace with your MSI username
LOCAL_DATASET="./SDOBenchmark-data-full.zip"  # Path to your local dataset zip
MSI_TRANSFER_NODE="login.msi.umn.edu"
MSI_SCRATCH_DIR="~/scratch/SDOModels/data/"

# Check if dataset exists locally
if [ ! -f "$LOCAL_DATASET" ]; then
    echo "Error: Dataset file $LOCAL_DATASET not found."
    exit 1
fi

# Create scratch directory on MSI
echo "Creating scratch directory on MSI..."
ssh $MSI_USERNAME@$MSI_TRANSFER_NODE "mkdir -p $MSI_SCRATCH_DIR"

# Transfer dataset to MSI
echo "Transferring dataset to MSI (this may take some time)..."
rsync -avP --progress $LOCAL_DATASET $MSI_USERNAME@$MSI_TRANSFER_NODE:$MSI_SCRATCH_DIR

# Extract dataset on MSI
echo "Extracting dataset on MSI..."
ssh $MSI_USERNAME@$MSI_TRANSFER_NODE "cd $MSI_SCRATCH_DIR && unzip -o SDOBenchmark-data-full.zip && echo 'Dataset extraction complete.'"

# Verify extraction
echo "Verifying dataset extraction..."
ssh $MSI_USERNAME@$MSI_TRANSFER_NODE "ls -la $MSI_SCRATCH_DIR/SDOBenchmark-data-full"

echo "Dataset transfer and preparation complete." 