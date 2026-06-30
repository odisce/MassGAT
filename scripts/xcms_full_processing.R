#!/usr/bin/env Rscript
library(xcms)
library(MSnbase)
library(BiocParallel)

register(SerialParam())

process_qehf_dataset <- function(input_folder, output_folder) {
    cat("--- Starting reproduction of Li et al. 2018 (Thermo QE HF) ---\n")
    
    ms_files <- list.files(input_folder, pattern = "\\.(mzML|mzXML)$", full.names = TRUE, ignore.case = TRUE)
    if (length(ms_files) == 0) stop("No files found in /data")
    
    raw_data <- readMSData(ms_files, mode = "onDisk")

    # 1. Feature Detection (centWave)
    cwp <- CentWaveParam(
        ppm = 5,                 
        peakwidth = c(5, 30),    
        snthresh = 10,           
        mzdiff = 0.005,          
        prefilter = c(4, 100),   
        noise = 0                
    )
    
    cat("Detecting peaks (QE HF settings)...\n")
    data_peaks <- findChromPeaks(raw_data, param = cwp)

    # 2. Alignment & Grouping
    pdp <- PeakDensityParam(
        sampleGroups = rep(1, length(ms_files)), 
        bw = 5,                  
        minFrac = 0.5            
    )
    
    cat("Grouping and Aligning...\n")
    data_aligned <- groupChromPeaks(data_peaks, param = pdp)
    
    # 3. Fill Missing Peaks 
    cat("Filling gaps...\n")
    data_filled <- fillChromPeaks(data_aligned)

    # 4. Extracting Consensus Features
    feature_table <- featureValues(data_filled, value = "into", method = "medret")
    
    # Filter 
    is_consensus <- apply(feature_table, 1, function(x) !any(is.na(x)))
    
    feature_metadata <- as.data.frame(featureDefinitions(data_filled)[is_consensus, ])
    feature_metadata <- feature_metadata[, !sapply(feature_metadata, is.list)] 
    
    consensus_intensities <- feature_table[is_consensus, ]
    final_output <- cbind(feature_metadata, consensus_intensities)

    # 5. Save results
    output_path <- file.path(output_folder, "XCMS_QEHF_Consensus.csv")
    write.csv(final_output, output_path, row.names = FALSE)
    
    cat("Success! QE HF results saved to:", output_path, "\n")
    cat("Total Consensus Features found:", nrow(final_output), "\n")
}

main <- function() {
    if (file.exists("/data") && file.exists("/output")) {
        input_folder <- "/data"
        output_folder <- "/output"
    } else {
        stop("Singularity mount points /data or /output not found.")
    }
    process_qehf_dataset(input_folder, output_folder)
}

if (!interactive()) { main() }