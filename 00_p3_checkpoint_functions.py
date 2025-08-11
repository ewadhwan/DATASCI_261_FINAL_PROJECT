# Databricks notebook source
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# COMMAND ----------

print(f"Current notebook path: {notebook_path}")

# COMMAND ----------

from datetime import datetime

# Configuration for checkpointing
section = "02"
number = "01"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}"
base_dir = "dbfs:/mnt/mids-w261/"
checkpoint_dir = f"{folder_path}/checkpoints"

# Utility functions for checkpointing
def save_checkpoint(df, checkpoint_name, description=""):
    """Save a dataframe checkpoint with metadata"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[{timestamp}] Saving checkpoint: {checkpoint_name}")
    if description:
        print(f"Description: {description}")
    
    # Save the dataframe
    df.coalesce(10).write.mode("overwrite").parquet(checkpoint_path)
    
    # Save metadata
    row_count = df.count()
    col_count = len(df.columns)
    
    print(f"✓ Checkpoint saved: {row_count:,} rows x {col_count} columns")
    print(f"  Path: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_name):
    """Load a dataframe from checkpoint"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[{timestamp}] Loading checkpoint: {checkpoint_name}")
    df = spark.read.parquet(checkpoint_path)
    
    row_count = df.count()
    col_count = len(df.columns)
    print(f"✓ Checkpoint loaded: {row_count:,} rows x {col_count} columns")
    
    return df

def checkpoint_exists(checkpoint_name):
    """Check if a checkpoint exists"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    try:
        # Try to list the path to see if it exists
        dbutils.fs.ls(checkpoint_path)
        return True
    except:
        return False
    
def list_checkpoints():
    """List all checkpoint folders in the checkpoint directory"""
    print(f"Listing checkpoints in: {checkpoint_dir}")
    try:
        files = dbutils.fs.ls(checkpoint_dir)
        checkpoint_names = [f.name.strip("/") for f in files if f.isDir()]
        if checkpoint_names:
            print("Available checkpoints:")
            for name in checkpoint_names:
                print(f"  • {name}")
        else:
            print("No checkpoints found.")
        return checkpoint_names
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []
    
def delete_checkpoint(checkpoint_name):
    """Delete a specific checkpoint folder"""
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
    print(f"Deleting checkpoint: {checkpoint_path}")
    try:
        dbutils.fs.rm(checkpoint_path, recurse=True)
        print("✓ Checkpoint deleted.")
    except Exception as e:
        print(f"Failed to delete checkpoint: {e}")