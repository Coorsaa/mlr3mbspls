#' @title Flip MB-sPLS Component Signs for Weight Alignment
#'
#' @description
#' This PipeOp flips the signs of MB-sPLS components to align a chosen reference
#' block's mean weight towards a specified direction (positive or negative).
#' 
#' Place this immediately after [PipeOpMBsPLS]. During training, it determines
#' per-component signs so that the reference block's mean weight points toward
#' the desired direction. It then flips all latent variable (LV) columns for
#' all blocks of the corresponding component. During prediction, it applies
#' the same signs determined during training.
#'
#' @details
#' The PipeOp can optionally flip the stored weights/loadings in the logging
#' environment (`log_env$mbspls_state`) so that any downstream code reading
#' the environment sees the aligned orientation.
#'
#' Sign determination priority:
#' 1. If `source = "weights"` and weights are available in `log_env`, use mean weights
#' 2. If weights unavailable, fall back to `source = "scores"` using mean training LV
#' 3. Components with zero or non-finite means default to positive orientation
#'
#' @param id Character string identifying this PipeOp. Defaults to "mbspls_flip".
#' @param param_vals Named list of parameter values. See parameters section.
#'
#' @section Parameters:
#' \describe{
#'   \item{ref_block}{`character(1)` \cr
#'     Name of the reference block used for sign determination. Required parameter.}
#'   \item{towards}{`factor` \cr
#'     Target orientation: "positive" (default) or "negative". Determines whether
#'     the reference block's mean weight should point toward positive or negative values.}
#'   \item{source}{`factor` \cr  
#'     Data source for sign determination: "weights" (default) or "scores".
#'     If "weights", uses `mean(weights[[ref_block]])` from log_env.
#'     If weights unavailable, automatically falls back to "scores".}
#'   \item{log_env}{`environment` or `NULL` \cr
#'     Same environment passed to `po("mbspls")`. Used to access weights
#'     and optionally store flipped weights/loadings. Defaults to NULL.}
#'   \item{flip_env_weights}{`logical(1)` \cr
#'     Whether to also flip weights/loadings stored in `log_env$mbspls_state`.
#'     Defaults to TRUE.}
#' }
#'
#' @return A [PipeOp] that modifies LV columns (LVk_<block>) with aligned signs.
#'   Only returns the LV columns that were flipped.
#'
#' @section State:
#' The `$state` after training contains:
#' \describe{
#'   \item{ref_block}{Reference block name used}
#'   \item{towards}{Target orientation used}  
#'   \item{signs}{Integer vector of signs (+1/-1) for each component}
#'   \item{K}{Number of components}
#'   \item{lv_map}{List mapping components to their LV column names per block}
#'   \item{blocks}{Character vector of block names}
#' }
#'
#' @examples
#' \dontrun{
#' # Basic usage after MB-sPLS
#' graph = po("mbspls", ncomp = 3) %>>%
#'         po("mbspls_flip", ref_block = "block1", towards = "positive")
#'
#' # With logging environment
#' log_env = new.env()
#' graph = po("mbspls", ncomp = 3, log_env = log_env) %>>%
#'         po("mbspls_flip", ref_block = "block1", log_env = log_env)
#' }
#'
#' @family PipeOps
#' @importFrom lgr lgr
#' @export
PipeOpMBsPLSFlipWeights = R6::R6Class(
    "PipeOpMBsPLSFlipWeights",
    inherit = mlr3pipelines::PipeOpTaskPreproc,
    public = list(
        
        #' @description
        #' Initialize the PipeOp.
        #' @param id Character string identifying this PipeOp.
        #' @param param_vals Named list of parameter values.
        initialize = function(id = "mbspls_flipweights", param_vals = list()) {
            ps <- paradox::ps(
                ref_block = paradox::p_uty(tags = "train", default = NULL),
                towards   = paradox::p_fct(levels = c("positive","negative"),
                                                                     default = "positive", tags = c("train","predict")),
                source    = paradox::p_fct(levels = c("weights","scores"),
                                                                     default = "weights",  tags = "train"),
                log_env   = paradox::p_uty(tags = c("train","predict"), default = NULL),
                flip_env_weights = paradox::p_lgl(default = TRUE, tags = "train")
            )
            super$initialize(id = id, param_set = ps, param_vals = param_vals)
            self$packages <- "mlr3mbspls"
            
            lgr$info("Initialized PipeOpMBsPLSFlipWeights with id '%s'", id)
        }
    ),
    private = list(

        .lv_column_map = function(dt_names) {
            # infer (k, block) from column names LV<k>_<block>
            lv_cols <- grep("^LV\\d+_", dt_names, value = TRUE)
            if (!length(lv_cols)) {
                lgr$warn("No LV columns found in data")
                return(list(K = 0L, blocks = character(0), map = list()))
            }
            
            comps  <- as.integer(sub("^LV(\\d+)_.*$", "\\1", lv_cols, perl = TRUE))
            blocks <- sub("^LV\\d+_", "", lv_cols)
            K      <- max(comps)
            bset   <- unique(blocks)
            
            lgr$info("Found %d components across %d blocks: %s", 
                             K, length(bset), paste(bset, collapse = ", "))
            
            # map[[k]] = named character vector of cols for that component (per block)
            map <- lapply(seq_len(K), function(k) {
                sel <- (comps == k)
                stats::setNames(lv_cols[sel], blocks[sel])
            })
            list(K = K, blocks = bset, map = map)
        },

        .get_signs_from_env = function(pv) {
            env <- pv$log_env
            if (!inherits(env, "environment")) {
                lgr$debug("No valid log_env provided")
                return(NULL)
            }
            
            st  <- env$mbspls_state
            if (is.null(st) || is.null(st$weights) || is.null(st$blocks) || is.null(st$ncomp)) {
                lgr$debug("log_env does not contain valid mbspls_state with weights")
                return(NULL)
            }
            
            lgr$info("Using weights from log_env for sign determination (%d components)", 
                             as.integer(st$ncomp))
            
            list(
                K      = as.integer(st$ncomp),
                blocks = names(st$blocks),
                W      = st$weights
            )
        },

        .decide_signs = function(ref_block, towards, pv, dt_names) {
            lgr$info("Determining component signs with ref_block='%s', towards='%s'", 
                             ref_block, towards)
            
            # Try weights first (log_env), else fall back to scores mean of LVk_<ref_block>
            st <- private$.get_signs_from_env(pv)
            if (!is.null(st) && ref_block %in% st$blocks && st$K > 0) {
                lgr$info("Using weights-based sign determination")
                s <- integer(st$K)
                for (k in seq_len(st$K)) {
                    m <- mean(as.numeric(st$W[[k]][[ref_block]]), na.rm = TRUE)
                    if (!is.finite(m) || m == 0) { 
                        s[k] <- +1L
                        lgr$debug("Component %d: zero/non-finite mean weight, defaulting to +1", k)
                    } else if (towards == "positive") { 
                        s[k] <- if (m >= 0) +1L else -1L
                        lgr$debug("Component %d: mean weight=%.3f, sign=%+d", k, m, s[k])
                    } else { 
                        s[k] <- if (m <= 0) +1L else -1L
                        lgr$debug("Component %d: mean weight=%.3f, sign=%+d", k, m, s[k])
                    }
                }
                lgr$info("Determined signs from weights: %s", paste(s, collapse = ", "))
                return(list(signs = s, K = st$K))
            }

            # scores fallback
            lgr$info("Falling back to scores-based sign determination")
            lm <- private$.lv_column_map(dt_names)
            if (!length(lm$K)) stop("Cannot infer MB-sPLS LV columns from task.")
            if (!(ref_block %in% lm$blocks))
                stop(sprintf("ref_block '%s' not found among LV columns (%s).",
                                         ref_block, paste(lm$blocks, collapse = ", ")))
            s <- integer(lm$K)
            for (k in seq_len(lm$K)) {
                col <- paste0("LV", k, "_", ref_block)
                m <- if (col %in% dt_names) mean(get(col, asNamespace("base"))(NULL)) else NA_real_
                # ^­­ (dummy to appease static checks) we'll compute properly below from dt
                s[k] <- +1L  # placeholder, real mean computed in .train_dt/.predict_dt
            }
            list(signs = s, K = lm$K, need_scores_mean = TRUE)
        },

        .flip_cols = function(dt, comp_signs, comp_map) {
            # comp_signs: integer K vector (+1/-1). comp_map[[k]]: named vector cols for blocks
            K <- length(comp_signs)
            flipped_count <- 0
            
            for (k in seq_len(K)) if (is.finite(comp_signs[k]) && comp_signs[k] == -1L) {
                cols_k <- unname(comp_map[[k]])
                cols_k <- cols_k[cols_k %in% names(dt)]
                if (length(cols_k)) {
                    lgr$debug("Flipping component %d columns: %s", k, paste(cols_k, collapse = ", "))
                    for (cn in cols_k) data.table::set(dt, j = cn, value = -dt[[cn]])
                    flipped_count <- flipped_count + length(cols_k)
                }
            }
            
            if (flipped_count > 0) {
                lgr$info("Flipped %d LV columns across %d components", flipped_count, sum(comp_signs == -1L))
            } else {
                lgr$info("No LV columns needed flipping (all components already aligned)")
            }
            
            dt
        },

        .flip_env_weights_if_any = function(pv, comp_signs) {
            if (!isTRUE(pv$flip_env_weights)) {
                lgr$debug("Skipping environment weights flipping (flip_env_weights=FALSE)")
                return(invisible(NULL))
            }
            
            env <- pv$log_env
            if (!inherits(env, "environment")) {
                lgr$debug("No valid log_env for weights flipping")
                return(invisible(NULL))
            }
            
            st  <- env$mbspls_state
            if (is.null(st) || is.null(st$weights) || is.null(st$loadings)) {
                lgr$debug("log_env missing weights/loadings for flipping")
                return(invisible(NULL))
            }
            
            K <- length(st$weights); bn <- names(st$blocks)
            flipped_comps <- 0

            for (k in seq_len(K)) {
                s <- as.integer(comp_signs[k] %||% +1L)
                if (s == 1L) next
                
                flipped_comps <- flipped_comps + 1
                lgr$debug("Flipping environment weights/loadings for component %d", k)
                
                for (b in bn) {
                    st$weights [[k]][[b]] <- s * st$weights [[k]][[b]]
                    st$loadings[[k]][[b]] <- s * st$loadings[[k]][[b]]
                }
            }
            
            # Optionally flip stored training scores if present
            if (!is.null(st$T_mat_train)) {
                lgr$debug("Flipping stored training scores in environment")
                for (k in seq_len(K)) {
                    s <- as.integer(comp_signs[k] %||% +1L); if (s == 1L) next
                    # flip all blocks' columns for component k
                    cols <- grep(sprintf("^LV%d_", k), colnames(st$T_mat_train), value = TRUE)
                    if (length(cols)) st$T_mat_train[, cols] <- s * st$T_mat_train[, cols, drop = FALSE]
                }
            }
            
            env$mbspls_state <- st
            
            if (flipped_comps > 0) {
                lgr$info("Flipped weights/loadings in log_env for %d components", flipped_comps)
            }
            
            invisible(NULL)
        },

        # ------------------ train: decide signs, flip LVs, store state -------------
        .train_dt = function(dt, levels, target = NULL) {
            lgr$info("Training PipeOpMBsPLSFlipWeights")
            
            pv <- utils::modifyList(
              paradox::default_values(self$param_set),
              self$param_set$get_values(tags = "train"),
              keep.null = TRUE
            )
            if (is.null(pv$ref_block) || !nzchar(pv$ref_block))
                stop("PipeOpMBsPLSFlip: please set param 'ref_block' to a block name.")

            # Map LV columns
            lm <- private$.lv_column_map(names(dt))
            if (lm$K == 0L)
                stop("PipeOpMBsPLSFlip: no LV columns (LV<k>_<block>) found. Place this after po('mbspls').")

            # Decide signs (prefer weights from env)
            ds <- private$.decide_signs(as.character(pv$ref_block), pv$towards, pv, names(dt))
            signs <- as.integer(ds$signs); K <- as.integer(ds$K)

            # If we fell back to scores, actually compute means now
            if (isTRUE(ds$need_scores_mean)) {
                lgr$debug("Computing component signs from training scores")
                for (k in seq_len(K)) {
                    col <- paste0("LV", k, "_", pv$ref_block)
                    if (!col %in% names(dt)) next
                    m <- mean(dt[[col]], na.rm = TRUE)
                    signs[k] <- if (pv$towards == "positive") { if (is.finite(m) && m < 0) -1L else +1L
                                         } else                           { if (is.finite(m) && m > 0) -1L else +1L }
                    lgr$debug("Component %d: mean score=%.3f, sign=%+d", k, m, signs[k])
                }
                lgr$info("Determined signs from scores: %s", paste(signs, collapse = ", "))
            }

            # Flip LV columns in training data (in place by returning replacements)
            dt_out <- data.table::as.data.table(dt[, unlist(lm$map, use.names = FALSE), with = FALSE])
            colnames(dt_out) <- unlist(lm$map, use.names = FALSE)
            dt_out <- private$.flip_cols(dt_out, signs, lm$map)

            # Store per-component signs for predict()
            self$state$ref_block <- as.character(pv$ref_block)
            self$state$towards   <- pv$towards
            self$state$signs     <- signs
            self$state$K         <- K
            self$state$lv_map    <- lm$map  # list(K): names=blocks, values=colnames
            self$state$blocks    <- unique(names(lm$map[[1]]))

            lgr$info("Stored state: ref_block='%s', K=%d, signs=[%s]", 
                             self$state$ref_block, self$state$K, paste(signs, collapse = ", "))

            # Optionally flip the weights/log_env copy too
            private$.flip_env_weights_if_any(pv, signs)

            lgr$info("Training completed successfully")
            dt_out
        },

        # ------------------ predict: reuse train-time signs, flip test -------------
        .predict_dt = function(dt, levels, target = NULL) {
            lgr$info("Flipping weights if needed...")

            st <- self$state
            if (is.null(st$signs) || is.null(st$lv_map))
                stop("PipeOpMBsPLSFlip: state not initialised; train() must run first.")
            
            # Build an output table with the LV cols we will replace
            cols <- unlist(st$lv_map, use.names = FALSE)
            cols <- cols[cols %in% names(dt)]
            if (!length(cols)) {
                lgr$warn("No LV columns found in prediction data")
                return(data.table::data.table())
            }
            
            lgr$debug("Applying stored signs to %d LV columns", length(cols))
            dt_out <- data.table::as.data.table(dt[, ..cols])
            colnames(dt_out) <- cols
            dt_out <- private$.flip_cols(dt_out, st$signs, st$lv_map)

            lgr$info("Flipping completed successfully")
            dt_out
        }
    )
)
