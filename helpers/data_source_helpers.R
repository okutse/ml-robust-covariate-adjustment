get_runtime_option <- function(option_names, default = "") {
  if (length(option_names) == 0) {
    return(default)
  }

  args <- commandArgs(trailingOnly = TRUE)
  for (option_name in option_names) {
    pattern <- paste0("^", gsub("([][{}()^$.|*+?\\\\-])", "\\\\1", option_name), "=")
    matched_arg <- args[grepl(pattern, args, ignore.case = TRUE)]
    if (length(matched_arg) > 0) {
      return(sub("^[^=]+=", "", matched_arg[1]))
    }
  }

  for (option_name in option_names) {
    env_value <- Sys.getenv(option_name, unset = NA_character_)
    if (!is.na(env_value) && nzchar(env_value)) {
      return(env_value)
    }
    upper_name <- toupper(option_name)
    env_value <- Sys.getenv(upper_name, unset = NA_character_)
    if (!is.na(env_value) && nzchar(env_value)) {
      return(env_value)
    }
  }

  default
}

normalize_data_source <- function(value, default = "local") {
  value <- tolower(trimws(ifelse(is.null(value), default, value)))
  if (!nzchar(value)) {
    value <- default
  }
  value
}

strip_doi_prefix <- function(reference) {
  reference <- trimws(reference)
  reference <- sub("^https?://(dx\\.)?doi\\.org/", "", reference, ignore.case = TRUE)
  reference <- sub("^doi:", "", reference, ignore.case = TRUE)
  reference
}

resolve_zenodo_download_url <- function(reference, preferred_file = NULL) {
  if (is.null(reference) || !nzchar(reference)) {
    return("")
  }

  reference <- trimws(reference)
  if (grepl("\\.zip(\\?.*)?$", reference, ignore.case = TRUE)) {
    return(reference)
  }
  if (grepl("^https?://", reference, ignore.case = TRUE) && !grepl("doi\\.org", reference, ignore.case = TRUE)) {
    return(reference)
  }

  record_id <- strip_doi_prefix(reference)
  record_id <- sub("^10\\.5281/zenodo\\.", "", record_id)
  if (!nzchar(record_id)) {
    return("")
  }

  api_url <- paste0("https://zenodo.org/api/records/", record_id)
  api_file <- tempfile(fileext = ".json")
  utils::download.file(api_url, api_file, mode = "wb", quiet = TRUE)
  record <- jsonlite::fromJSON(api_file)

  if (is.null(record$files) || length(record$files) == 0) {
    stop(paste0("No downloadable files were found for Zenodo record ", record_id, "."))
  }

  files <- as.data.frame(record$files, stringsAsFactors = FALSE)
  file_names <- if ("key" %in% names(files)) files$key else if ("filename" %in% names(files)) files$filename else character(0)
  if (length(file_names) == 0) {
    stop(paste0("Unable to determine file names for Zenodo record ", record_id, "."))
  }

  file_links <- NULL
  if ("links" %in% names(files)) {
    if (is.data.frame(files$links) && "self" %in% names(files$links)) {
      file_links <- files$links$self
    } else if (is.list(files$links) && !is.null(files$links$self)) {
      file_links <- files$links$self
    }
  } else if ("links.self" %in% names(files)) {
    file_links <- files$links.self
  } else if ("links.download" %in% names(files)) {
    file_links <- files$links.download
  }

  if (!is.null(preferred_file) && nzchar(preferred_file)) {
    preferred_matches <- which(basename(file_names) == preferred_file)
    if (length(preferred_matches) > 0) {
      if (!is.null(file_links)) {
        return(as.character(file_links[preferred_matches[1]]))
      }
      stop(paste0("Zenodo file link for ", preferred_file, " was not exposed in the record metadata."))
    }
  }

  zip_matches <- which(grepl("\\.zip$", basename(file_names), ignore.case = TRUE))
  if (length(zip_matches) > 0) {
    if (!is.null(file_links)) {
      return(as.character(file_links[zip_matches[1]]))
    }
    stop("Zenodo zip file link was not exposed in the record metadata.")
  }

  if (!is.null(file_links)) {
    return(as.character(file_links[1]))
  }
  stop("Zenodo record metadata did not expose a downloadable file link.")
}

safe_unzip_archive <- function(zip_path, dest_dir) {
  if (!file.exists(zip_path)) {
    stop(paste("Archive file not found:", zip_path))
  }
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  }

  archive_listing <- utils::unzip(zip_path, list = TRUE)
  archive_names <- if ("Name" %in% names(archive_listing)) archive_listing$Name else character(0)
  if (length(archive_names) == 0) {
    stop(paste("Archive contains no files:", zip_path))
  }
  unsafe_entries <- archive_names[grepl("(^/|^[A-Za-z]:|\\.\\./)", archive_names)]
  if (length(unsafe_entries) > 0) {
    stop(paste("Archive contains unsafe paths:", paste(unsafe_entries, collapse = ", ")))
  }

  utils::unzip(zip_path, exdir = dest_dir)
  invisible(dest_dir)
}

download_archive_file <- function(url, zip_path, timeout_secs = 3600) {
  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  if (is.numeric(timeout_secs) && is.finite(timeout_secs) && timeout_secs > old_timeout) {
    options(timeout = timeout_secs)
  }
  utils::download.file(url, zip_path, mode = "wb", quiet = FALSE)
  invisible(zip_path)
}

find_archive_root <- function(base_dir, file_pattern, parent_levels = 0) {
  if (is.null(base_dir) || !nzchar(base_dir) || !dir.exists(base_dir)) {
    stop(paste("Archive directory does not exist:", base_dir))
  }

  archive_files <- list.files(base_dir, pattern = file_pattern, full.names = TRUE, recursive = TRUE)
  if (length(archive_files) == 0) {
    stop(paste("No matching files found in archive directory:", base_dir))
  }

  root_dir <- dirname(archive_files[1])
  if (parent_levels > 0) {
    for (idx in seq_len(parent_levels)) {
      root_dir <- dirname(root_dir)
    }
  }

  root_dir
}

copy_matching_files <- function(source_dir, dest_dir, file_pattern) {
  if (!dir.exists(source_dir)) {
    stop(paste("Source directory does not exist:", source_dir))
  }
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  }
  matching_files <- list.files(source_dir, pattern = file_pattern, full.names = TRUE, recursive = TRUE)
  if (length(matching_files) == 0) {
    stop(paste("No matching files found in:", source_dir))
  }
  file.copy(matching_files, dest_dir, overwrite = TRUE)
  invisible(dest_dir)
}

local_dataset_available <- function(dataset_dir, file_pattern) {
  dir.exists(dataset_dir) && length(list.files(dataset_dir, pattern = file_pattern, full.names = TRUE)) > 0
}
