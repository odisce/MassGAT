#!/usr/bin/env Rscript
library(xcms)
library(MSnbase)
library(BiocParallel)

register(SerialParam())

process_qehf_dataset <- function(input_folder, output_folder) {
    cat("--- Starting alignment with MassGAT results ---\n")
    
    ms_files <- list.files(input_folder, pattern = "\\.(mzML|mzXML)$", full.names = TRUE, ignore.case = TRUE)
    if (length(ms_files) == 0) stop("No files found in /data")
    
    # 1. Load the raw data
    raw_data <- readMSData(ms_files, mode = "onDisk")

    # 2. Convert to XCMSnExp object
    xdata <- as(raw_data, "XCMSnExp")

    # 3. Feature Detection
    file_path <- "/workspace/chroms_gnn/results/predictions_qehf_peaks_metadata.tsv"
    if (grepl("\\.csv$", file_path, ignore.case = TRUE)) {
        custom_peaks <- read.csv(file_path)
    } else if (grepl("\\.tsv$", file_path, ignore.case = TRUE)) {
        custom_peaks <- read.delim(file_path, sep = "\t")
    } else {
        stop("File extension not recognized. Please use .csv or .tsv files.")
    }

    required_cols <- c("mz", "mzmin", "mzmax", "rt", "rtmin", "rtmax", "into", "sample")
    if (!all(required_cols %in% colnames(custom_peaks))) {
        stop("Missing required columns: ", paste(setdiff(required_cols, colnames(custom_peaks)), collapse = ", "))
    }

    custom_peaks_matrix <- as.matrix(custom_peaks[, required_cols])

    # Ensure all columns are numeric
    if (!is.numeric(custom_peaks_matrix)) {
        custom_peaks_matrix <- apply(custom_peaks_matrix, 2, as.numeric)
        custom_peaks_matrix <- custom_peaks_matrix[complete.cases(custom_peaks_matrix), ]
    }

    rownames(custom_peaks_matrix) <- NULL

    if (!is.numeric(custom_peaks_matrix)) {
    stop("Matrix contains non-numeric values after conversion")
}

    chromPeaks(xdata) <- custom_peaks_matrix

    # 4. Alignment
    pdp <- PeakDensityParam(
        sampleGroups = rep(1, length(ms_files)), 
        bw = 5,                  
        binSize = 0.005,         
        minFrac = 0.5            
    )

    cat("Grouping and Aligning...\n")
    data_aligned <- groupChromPeaks(xdata, param = pdp)

    # 3. Fill Missing Peaks 
    cat("Filling gaps...\n")
    data_filled <- fillChromPeaks(data_aligned)

    # 4. Extracting Consensus Features
    feature_table <- featureValues(data_filled, value = "into", method = "medret")
    
    is_consensus <- apply(feature_table, 1, function(x) !any(is.na(x)))
    
    feature_metadata <- as.data.frame(featureDefinitions(data_filled)[is_consensus, ])
    feature_metadata <- feature_metadata[, !sapply(feature_metadata, is.list)] 
    
    consensus_intensities <- feature_table[is_consensus, ]
    final_output <- cbind(feature_metadata, consensus_intensities)

    # 5. Save results
    output_path <- file.path(output_folder, "qehf_peaks_metadata_aligned.csv")
    write.csv(final_output, output_path, row.names = FALSE)
    
    cat("Success! QEHF MassGAT Final results saved to:", output_path, "\n")
    cat("Total Consensus Features found:", nrow(final_output), "\n")
}


main <- function() {
    if (file.exists("/data") && file.exists("/output") && file.exists("/workspace")) {
        input_folder <- "/data"
        output_folder <- "/output"
    } else {
        stop("Singularity mount points /data or /output not found.")
    }
    process_qehf_dataset(input_folder, output_folder)
}

if (!interactive()) { main() }