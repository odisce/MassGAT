#!/usr/bin/env Rscript
library(ionshed)
library(arrow)
library(MSnbase)
library(yaml)

# Works whether the script is run via `Rscript file.R` or `source(file.R)`
get_script_path <- function() {
    cmd_args <- commandArgs(trailingOnly = FALSE)
    file_flag <- "--file="
    match <- grep(file_flag, cmd_args)
    if (length(match) > 0) {
        return(normalizePath(sub(file_flag, "", cmd_args[match[1]])))
    }
    if (!is.null(sys.frame(1)$ofile)) {
        return(normalizePath(sys.frame(1)$ofile))
    }
    stop("Could not determine script path; pass config_path explicitly to load_config().")
}

# Load configuration from YAML
load_config <- function(config_path = NULL) {
    if (is.null(config_path) && nzchar(Sys.getenv("MASSGAT_CONFIG"))) {
        config_path <- Sys.getenv("MASSGAT_CONFIG")
    }
    if (is.null(config_path)) {
        # Try to find config.yaml relative to this script (project_root/config.yaml,
        # assuming script lives at project_root/scripts/)
        script_dir <- dirname(dirname(get_script_path()))
        config_path <- file.path(script_dir, 'config.yaml')
    }
    
    if (!file.exists(config_path)) {
        stop("Configuration file not found: ", config_path)
    }
    
    config <- yaml::read_yaml(config_path)
    return(config)
}

save_config_metadata <- function(output_file, config, relevant_keys) {
    metadata_file <- sub("\\.feather$", ".config.json", output_file)
    relevant_config <- config$peak_picking[relevant_keys]
    relevant_config <- Filter(Negate(is.null), relevant_config)
    writeLines(jsonlite::toJSON(relevant_config, pretty = TRUE, digits = NA), metadata_file)
}

config_changed <- function(output_file, config, relevant_keys) {
    metadata_file <- sub("\\.feather$", ".config.json", output_file)
    if (!file.exists(metadata_file)) {
        return(TRUE)
    }
    tryCatch({
        stored_config <- jsonlite::fromJSON(readLines(metadata_file))
        current_config <- config$peak_picking[relevant_keys]
        current_config <- Filter(Negate(is.null), current_config)
        # Through JSON as the stored settings, so that 6 and 6.0 compare equal
        current_config <- jsonlite::fromJSON(jsonlite::toJSON(current_config, digits = NA))
        return(!identical(stored_config, current_config))
    }, error = function(e) {
        return(TRUE)
    })
}

# Load configuration
config <- load_config()
RTTOL <- config$peak_picking$rttol
PPM <- config$peak_picking$ppm
MZ_RES <- config$peak_picking$mz_res
RT_RES <- config$peak_picking$rt_res
XIC_RTTOL <- config$peak_picking$xic_rttol

cat("Loaded configuration:\n")
cat("  RTTOL:", RTTOL, "\n")
cat("  PPM:", PPM, "\n")
cat("  MZ_RES:", MZ_RES, "\n")
cat("  RT_RES:", RT_RES, "\n")
cat("  XIC_RTTOL:", XIC_RTTOL, "\n")


# Function to process a single mzML/mzXML file
process_file <- function(file_path, output_folder, force = FALSE, config = NULL) {
    # Check if folder exists and create it if not
    if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
    }
    
    cat("Processing file:", file_path, "\n")
    time1 <- Sys.time()
    
    # Get basename for output files (handle both mzML and mzXML)
    base_path <- basename(file_path)
    base_path <- sub("\\.(mzML|mzXML)$", "", base_path, ignore.case = TRUE)
    
    # Check if output files already exist (skip if they do)
    peak_output <- file.path(output_folder, paste0(base_path, "_peak.feather"))
    roi_output <- file.path(output_folder, paste0(base_path, "_signal.feather"))
    
    skip <- file.exists(peak_output) && file.exists(roi_output) && !force
    if (skip && !is.null(config) && config_changed(peak_output, config, names(config$peak_picking))) {
        cat("CONFIG CHANGED: Reprocessing", base_path, "\n")
        skip <- FALSE
    }
    
    if (skip) {
        cat("Output files already exist for", base_path, "- skipping\n")
        return(list(status = "skipped", file = base_path))
    }
    
    tryCatch({
        # Read the mzML/mzXML file as an MSnExp object
        cat("Reading MS file...\n")
        MSnExpObj <- readMSData(file_path, mode = "onDisk", msLevel. = 1L)
        cat("MSnExp object created\n")
        
        # Compute the peak list and the signals
        cat("Computing peak list...\n")
        res <- ionshed(MSnExpObj, rttol = RTTOL, ppm = PPM, mz_res = MZ_RES, rt_res = RT_RES)
        cat("Peak list computed\n")

        if (!is.null(res$peak_info)) {
            cat(paste("nrow(res$peak_info):", nrow(res$peak_info)), "\n")
        }
        if (!is.null(res$roi_data)) {
            cat(paste("nrow(res$roi_data):", nrow(res$roi_data)), "\n")
        }
        # Sort the signals by ROI before saving them
        data.table::setkey(res$roi_data, "roi")
        cat("Signals sorted by ROI\n")

        # Save results in arrow format
        cat("Saving results...\n")
        write_feather(res$peak_info, peak_output)
        write_feather(res$roi_data, roi_output)
        if (!is.null(config)) {
            save_config_metadata(peak_output, config, names(config$peak_picking))
        }
        
        time2 <- Sys.time()
        processing_time <- difftime(time2, time1, units = "mins")
        
        cat("Successfully processed", base_path, "in", round(processing_time, 2), "minutes\n")
        cat("Output files saved:\n")
        cat("  Peak info:", peak_output, "\n")
        cat("  ROI data:", roi_output, "\n")
        
        # Clean up memory
        rm(MSnExpObj, res)
        gc()
        
        return(list(
            status = "success", 
            file = base_path, 
            processing_time = processing_time,
            peak_output = peak_output,
            roi_output = roi_output
        ))
        
    }, error = function(e) {
        cat("ERROR processing", base_path, ":", conditionMessage(e), "\n")
        return(list(status = "error", file = base_path, error = conditionMessage(e)))
    })
}

# process_file() run in a worker process, its messages returned with the result.
# One BiocParallel worker per process: by default MSnbase would start one worker per core of the machine for each file.
process_file_logged <- function(file_path, output_folder, force) {
    BiocParallel::register(BiocParallel::SerialParam())
    log_lines <- utils::capture.output(
        result <- process_file(file_path, output_folder, force = force, config = config)
    )
    result$log <- log_lines
    result
}

# Process the files in parallel, one file per worker process (fork on Linux/macOS, PSOCK cluster on Windows).
# Results come back in the order of ms_files; the messages of each file are printed when all files are done.
process_files_parallel <- function(ms_files, output_folder, force, cores) {
    cat("Processing", length(ms_files), "files with", cores, "worker processes\n")
    if (.Platform$OS.type == "windows") {
        cl <- parallel::makePSOCKcluster(cores)
        on.exit(parallel::stopCluster(cl), add = TRUE)
        parallel::clusterEvalQ(cl, {
            suppressPackageStartupMessages({
                library(ionshed)
                library(arrow)
                library(MSnbase)
                library(yaml)
            })
            NULL
        })
        parallel::clusterExport(cl, c("process_file", "process_file_logged", "config_changed", "save_config_metadata",
                                      "config", "RTTOL", "PPM", "MZ_RES", "RT_RES", "XIC_RTTOL"), envir = globalenv())
        results <- parallel::parLapply(cl, ms_files, process_file_logged, output_folder, force)
    } else {
        results <- parallel::mclapply(ms_files, process_file_logged, output_folder, force,
                                      mc.cores = cores, mc.preschedule = FALSE)
    }
    for (i in seq_along(ms_files)) {
        result <- results[[i]]
        if (!is.list(result) || is.null(result$status)) {
            # The worker itself failed (e.g. killed when out of memory)
            result <- list(status = "error", file = sub("\\.(mzML|mzXML)$", "", basename(ms_files[i]), ignore.case = TRUE),
                           error = paste(as.character(result), collapse = " "))
            results[[i]] <- result
        }
        cat("\n--- File", i, "of", length(ms_files), "---\n")
        cat("File:", basename(ms_files[i]), "\n")
        if (length(result$log) > 0) {
            cat(result$log, sep = "\n")
        }
        cat(switch(result$status, success = "SUCCESS", skipped = "SKIPPED", "ERROR"), "\n", sep = "")
    }
    results
}

# Function to process all mzML/mzXML files in a directory
process_all_files <- function(input_folder, output_folder, force = FALSE, cores = 1) {
    cat("=======================================================\n")
    cat("Ionshed Batch Processing\n")
    cat("=======================================================\n")
    cat("Input folder:", input_folder, "\n")
    cat("Output folder:", output_folder, "\n")
    
    # Check if input folder exists
    if (!dir.exists(input_folder)) {
        stop("Input folder does not exist: ", input_folder)
    }
    
    # Create output folder if it doesn't exist
    if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
        cat("Created output folder:", output_folder, "\n")
    }
    
    # Find all mzML and mzXML files
    ms_files <- list.files(input_folder, pattern = "\\.(mzML|mzXML)$", full.names = TRUE, ignore.case = TRUE)
    
    if (length(ms_files) == 0) {
        stop("No mzML or mzXML files found in input folder: ", input_folder)
    }
    
    cat("Found", length(ms_files), "MS files to process\n")
    
    # Print file breakdown by type
    mzml_count <- sum(grepl("\\.mzML$", ms_files, ignore.case = TRUE))
    mzxml_count <- sum(grepl("\\.mzXML$", ms_files, ignore.case = TRUE))
    cat("  - mzML files:", mzml_count, "\n")
    cat("  - mzXML files:", mzxml_count, "\n")
    cat("=======================================================\n")
    
    # Initialize results tracking
    results <- list()
    start_time <- Sys.time()
    
    cores <- max(1L, min(as.integer(cores), parallel::detectCores() - 1L, length(ms_files), na.rm = TRUE))
    # Process each file (one file per worker process when cores > 1)
    if (cores > 1) {
        results <- process_files_parallel(ms_files, output_folder, force, cores)
    } else for (i in seq_along(ms_files)) {
        file_path <- ms_files[i]
        
        cat("\n--- Processing file", i, "of", length(ms_files), "---\n")
        cat("File:", basename(file_path), "\n")
        
        # Process the file
        result <- process_file(file_path, output_folder, force = force, config = config)
        results[[i]] <- result
        
        # Print progress
        if (result$status == "success") {
            cat("SUCCESS\n")
        } else if (result$status == "skipped") {
            cat("SKIPPED\n")
        } else {
            cat("ERROR\n")
        }
    }
    
    # Summary
    end_time <- Sys.time()
    total_time <- difftime(end_time, start_time, units = "mins")
    
    success_count <- sum(sapply(results, function(x) x$status == "success"))
    error_count <- sum(sapply(results, function(x) x$status == "error"))
    skipped_count <- sum(sapply(results, function(x) x$status == "skipped"))
    
    cat("\n=======================================================\n")
    cat("PROCESSING SUMMARY\n")
    cat("=======================================================\n")
    cat("Total files:", length(ms_files), "\n")
    cat("Successful:", success_count, "\n")
    cat("Errors:", error_count, "\n")
    cat("Skipped:", skipped_count, "\n")
    cat("Total processing time:", round(total_time, 2), "minutes\n")
    
    # Save processing log
    log_file <- file.path(output_folder, paste0("processing_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))
    
    log_content <- c(
        "Ionshed Processing Log",
        paste("Date:", Sys.time()),
        paste("Input folder:", input_folder),
        paste("Output folder:", output_folder),
        paste("Total files:", length(ms_files)),
        paste("Successful:", success_count),
        paste("Errors:", error_count),
        paste("Skipped:", skipped_count),
        paste("Total time:", round(total_time, 2), "minutes"),
        "",
        "File Details:"
    )
    
    for (result in results) {
        if (result$status == "success") {
            log_content <- c(log_content, paste("SUCCESS:", result$file, 
                           "(" , round(result$processing_time, 2), "min)"))
        } else if (result$status == "error") {
            log_content <- c(log_content, paste("ERROR:", result$file, "-", result$error))
        } else {
            log_content <- c(log_content, paste("SKIPPED:", result$file))
        }
    }
    
    writeLines(log_content, log_file)
    cat("Processing log saved to:", log_file, "\n")
    
    return(results)
}

# Main execution
main <- function() {
    # 1. Capture command-line arguments
    args <- commandArgs(trailingOnly = TRUE)
    
    force <- FALSE
    # Check for --force flag
    if ("--force" %in% args) {
        force <- TRUE
        args <- args[args != "--force"]
    }
    
    # --cores N (or --cores=N): number of files processed in parallel
    cores <- 1L
    i_cores <- grep("^--cores", args)
    if (length(i_cores) > 0) {
        value <- sub("^--cores=?", "", args[i_cores[1]])
        drop <- i_cores[1]
        if (!nzchar(value)) {
            value <- args[i_cores[1] + 1]
            drop <- c(drop, i_cores[1] + 1)
        }
        cores <- suppressWarnings(as.integer(value))
        if (is.na(cores) || cores < 1) {
            cat("ERROR: --cores needs a positive integer.\n")
            quit(status = 1)
        }
        args <- args[-drop]
    }
    
    # 2. Check if running in container environment or via CLI args
    if (file.exists("/workspace") && file.exists("/data")) {
        cat("Running in Singularity container environment\n")
        input_folder <- "/data"
        output_folder <- "/output"
    } else if (length(args) >= 2) {
        input_folder <- args[1]
        output_folder <- args[2]
    } else {
        cat("ERROR: Missing arguments.\n")
        cat("Usage: Rscript ionshed_batch_processing.R <input_folder> <output_folder> [--force] [--cores N]\n")
        quit(status = 1)
    }
    
    # Create output folder if it doesn't exist
    if (!dir.exists(output_folder)) {
        dir.create(output_folder, recursive = TRUE)
    }
    
    # Set up logging in output folder
    log_file <- file.path(output_folder, paste0("ionshed_processing_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log"))
    sink(log_file, split = TRUE)
    
    exit_code <- 0
    tryCatch({
        # Process all files
        results <- process_all_files(input_folder, output_folder, force = force, cores = cores)
        
        # Exit with appropriate code
        error_count <- sum(sapply(results, function(x) x$status == "error"))
        if (error_count > 0) {
            cat("Completed with", error_count, "errors\n")
            exit_code <- 1
        } else {
            cat("All files processed successfully!\n")
        }
        
    }, error = function(e) {
        # Catch fatal errors, turn off sink, and print to terminal
        sink() 
        cat("\n[FATAL ERROR]:", conditionMessage(e), "\n")
        cat("Check your file paths and ensure .mzML files exist in the input directory.\n")
        quit(status = 1)
        
    }, finally = {
        if (sink.number() > 0) {
            sink()
        }
        cat("Log saved to:", log_file, "\n")
    })
    
    quit(status = exit_code)
}

# Run main function
if (!interactive()) {
    main()
}