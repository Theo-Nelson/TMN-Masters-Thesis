#!/bin/bash
#SBATCH --job-name=WT_D1_visium_hd_analysis   # Job name
#SBATCH --output=WT_D1_visium_hd_analysis.out # Standard output log
#SBATCH --error=WT_D1_visium_hd_analysis.err  # Error log
#SBATCH --time=36:00:00                 # Wall time limit
#SBATCH --ntasks=16                     # Number of tasks (adjust as needed)
#SBATCH --partition=icelake-himem             # Partition name
#SBATCH --account=TEICHLAB-SL2-CPU       # Account name

# Define paths
WORKDIR="/home/tmn37/rds/rds-teichlab-C9woKbOCf2Y/tmn37/visiumHD_vsnl1_3D_reconstruction/WT_D1"
INPUT_DATA="/home/tmn37/rfs/rfs-teichlab-nfs-iCNyzSAaucw/tmn37/vsnl1-heart-3D-reconstruction"
OUTPUT_DIR="${WORKDIR}/output"
REF_DIR="/home/tmn37/rds/hpc-work/intestine_Visium_HD/references/refdata-gex-mm10-2020-A"
SPACERANGER_BIN="/home/tmn37/rds/hpc-work/intestine_Visium_HD/spaceranger/spaceranger-3.1.3/bin/spaceranger"

# Move to output directory
cd $OUTPUT_DIR

# Run Space Ranger count
$SPACERANGER_BIN count --id=visium_hd_count \
    --transcriptome=$REF_DIR \
    --fastqs=$INPUT_DATA/WT_D1_fastqs \
    --probe-set=$INPUT_DATA/Visium_Mouse_Transcriptome_Probe_Set_v2.0_mm10-2020-A.csv \
    --slide=H1-VRWJ69B \
    --area=D1 \
    --cytaimage=$INPUT_DATA/images/WT/CAVG10386_2024-12-04_12-19-07_2024-12-04_11-39-52_H1-VRWJ69B_D1_E3-slide1.tif \
    --image=$INPUT_DATA/images/WT/VisiumHD_007E3_slide1_34-35-36-37.btf \
    --create-bam=true \
    --jobmode=local \
    --localcores=8 \
    --localmem=120

echo "Visium HD analysis completed successfully."

