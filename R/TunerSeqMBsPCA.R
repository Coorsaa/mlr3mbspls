#' Sequential Component-wise Tuner for Group-Sparse MB-sPCA
#'
#' @description
#' `TunerSeqMBsPCA` tunes one component at a time for `PipeOpMBsPCA`, choosing
#' a block-sparsity vector **c** that maximises **cross-validated total variance
#' explained**. After each component it refits on the full residuals, performs
#' an optional permutation test for significance, and—if significant—deflates
#' all blocks before proceeding to the next component.
#'
#' The public interface mirrors `TunerSeqMBsPLS`; only the objective (variance
#' explained rather than latent correlation) and the C++ back-end differ.
#'
#' @section Works with:
#' A learner whose pipeline contains a `PipeOpMBsPCA` named `"mbspca"`. The
#' tuner writes the chosen √(L¹) budgets into `param_vals$c_matrix`
#' (`blocks × components`) of that pipeop.
#'
#' @section Construction:
#' `TunerSeqMBsPCA$new(tuner = "random_search", budget = 100L,`
#' `resampling = mlr3::rsmp("cv", folds = 3), parallel = "none",`
#' `early_stopping = TRUE, n_perm = 1000L, perm_alpha = 0.05)`
#'
#' @param tuner (`character(1)`) Optimizer ID for the inner single-component
#'   search (e.g. `"random_search"`). Asynchronous optimizers are coerced to
#'   `"random_search"`.
#' @param budget (`integer(1)`) Number of evaluations for each component-wise
#'   search.
#' @param resampling (`Resampling`) Template for inner CV (default 3-fold).
#' @param parallel (`character(1)`) `"none"` or `"inner"`; with `"inner"` the
#'   fold evaluations are parallelised via **future**.
#' @param early_stopping (`logical(1)`) Perform permutation test after each
#'   component and stop if not significant (PC-1 is always kept).
#' @param n_perm (`integer(1)`) Number of permutations for the test.
#' @param perm_alpha (`numeric(1)` in `[0,1]`) α-level for the test.
#'
#' @section Optimisation procedure:
#' For component *k*, a search space over integer proxies `c_b ∈ [1, p_b]` is
#' defined per block (*p_b* = #features). The objective evaluates CV mean of
#' total variance explained. The optimal integers are mapped to √(L¹) budgets
#' via `sqrt(c_b)` and stored in column *k* of `c_matrix`. Residuals are
#' deflated and the procedure continues for the next component (up to the
#' requested maximum, possibly shortened by early stopping).
#'
#' @return
#' A tuned `TuningInstance` (invisibly). The result is written to the instance
#' via `assign_result()` with `learner_param_vals = list(c_matrix = <matrix>)`
#' and the best inner objective value under the key `"mbspca.mean_ev"`.
#'
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3pipelines)
#' library(mlr3tuning)
#' blocks = list(eng = c("disp", "hp", "drat"), body = c("wt", "qsec"))
#' po = PipeOpMBsPCA$new(blocks = blocks, param_vals = list(ncomp = 3))
#' lrn = as_learner(po %>>% po("regr.ranger"))
#'
#' ti = TuningInstanceSingleCrit$new(
#'   task = tsk("mtcars"),
#'   learner = lrn,
#'   resampling = rsmp("holdout"),
#'   measure = msr("regr.rmse"),
#'   search_space = ps(), # outer space unused; tuner overrides mbspca internals
#'   terminator = trm("none")
#' )
#' TunerSeqMBsPCA$new(budget = 50)$optimize(ti)
#' ti$result_learner_param_vals$c_matrix
#' }
#'
#' @seealso [PipeOpMBsPCA], [mlr3tuning::Tuner], [bbotk]
#' @family mb-sPCA
#' @importFrom rlang "%||%"
#' @importFrom mlr3 rsmp
#' @importFrom data.table last
#' @import lgr
#' @export
TunerSeqMBsPCA = R6::R6Class(
  "TunerSeqMBsPCA",
  inherit = mlr3tuning::Tuner,

  public = list(
    #' @description Create a new TunerSeqMBsPCA.
    #' @param tuner character(1). Inner optimizer ID (e.g., "random_search").
    #' @param budget integer(1). Number of evaluations per component (default 100).
    #' @param resampling Resampling. Template for inner CV (default `rsmp("cv", folds = 3)`).
    #' @param parallel character(1). "none" or "inner" (future-based parallel fold evals).
    #' @param early_stopping logical(1). Enable permutation early stopping (default TRUE).
    #' @param n_perm integer(1). Number of permutations if early stopping is enabled.
    #' @param perm_alpha numeric(1). Significance level for permutation test.
    initialize = function(tuner = "random_search",
      budget = 100L,
      resampling = rsmp("cv", folds = 3),
      parallel = "none",
      early_stopping = TRUE,
      n_perm = 1000L,
      perm_alpha = 0.05) {

      checkmate::assert_choice(parallel, c("none", "inner"))
      checkmate::assert_int(budget, lower = 1L)
      checkmate::assert_flag(early_stopping)
      checkmate::assert_int(n_perm, lower = 1L)
      checkmate::assert_number(perm_alpha, lower = 0, upper = 1)

      if (grepl("async", tuner, ignore.case = TRUE)) {
        lgr$warning("Asynchronous tuners are unsupported – using 'random_search' instead.")
        tuner = "random_search"
      }

      private$.tuner = tuner
      private$.budget = budget
      private$.resampling_tpl = resampling
      private$.parallel = parallel
      private$.early_stop = early_stopping
      private$.n_perm = n_perm
      private$.perm_alpha = perm_alpha

      super$initialize(
        param_set = paradox::ps(),
        properties = "single-crit",
        param_classes = c("ParamInt", "ParamDbl", "ParamLgl", "ParamFct")
      )
    },
    #' @description Run the sequential tuning loop.
    #' @param instance [mlr3tuning::TuningInstanceSingleCrit] (or compatible). The outer instance to populate.
    #' @return The modified instance (invisibly).
    optimize = function(instance) private$.run(instance)
  ),

  private = list(

    # ───────────────────────── helpers ───────────────────────────────
    .pre_graph_before_mbspca = function(learner) {
      ids = learner$graph$ids()
      pos = match("mbspca", ids)
      if (is.na(pos) || pos == 1L) {
        return(mlr3pipelines::Graph$new())
      }
      mlr3pipelines::as_graph(
        learner$graph$clone(deep = TRUE)$pipeops[ids[seq_len(pos - 1L)]]
      )
    },

    .make_blocks = function(df, block_map) {
      lapply(names(block_map), function(bn) {
        cols = block_map[[bn]]
        if (!length(cols)) stop("Block '", bn, "' is empty.")
        as.matrix(df[, ..cols])
      })
    },

    .deflate_blocks = function(X, W) {
      B = length(X)
      for (b in seq_len(B)) {
        t_b = X[[b]] %*% W[[b]]
        n = drop(crossprod(t_b))
        if (n > 1e-10) {
          p_b = crossprod(X[[b]], t_b) / n
          X[[b]] = X[[b]] - tcrossprod(t_b, p_b)
        }
      }
      X
    },

    # ───────────────────────── main loop ─────────────────────────────
    .run = function(inst) {

      learner_tpl = inst$objective$learner
      task_full = inst$objective$task$clone(deep = TRUE)

      ## preprocessing graph (everything *before* mbspca)
      pre_graph_tpl = private$.pre_graph_before_mbspca(learner_tpl)

      ## blocks & maximum number of components requested
      blocks = learner_tpl$graph$pipeops$mbspca$blocks
      K_max = learner_tpl$graph$pipeops$mbspca$param_set$values$ncomp %||% 1L
      B = length(blocks)
      if (B == 0) stop("PipeOpMBsPCA has no blocks defined.")

      ## ---------- preprocess full data once -------------------------
      if (length(pre_graph_tpl$pipeops)) {
        pre_graph_full = pre_graph_tpl$clone(deep = TRUE)
        pre_graph_full$train(task_full)
        df_full = last(pre_graph_full$predict(task_full))$data()
      } else {
        df_full = task_full$data()
      }
      X_residual = private$.make_blocks(df_full, blocks)
      names(X_residual) = names(blocks)

      ## ---------- instantiate CV -----------------------------------
      rs = private$.resampling_tpl$clone()
      if (!rs$is_instantiated) rs$instantiate(task_full)

      fold_tr = vector("list", rs$iters)
      fold_val = vector("list", rs$iters)

      for (f in seq_len(rs$iters)) {
        task_tr = task_full$clone(deep = FALSE)$filter(rs$train_set(f))
        task_va = task_full$clone(deep = FALSE)$filter(rs$test_set(f))

        g = pre_graph_tpl$clone(deep = TRUE)
        df_tr = g$train(task_tr)[[1]]$data()
        df_va = g$predict(task_va)[[1]]$data()

        fold_tr[[f]] = private$.make_blocks(df_tr, blocks)
        fold_val[[f]] = private$.make_blocks(df_va, blocks)
      }

      ## ---------- optional parallelisation --------------------------
      if (private$.parallel == "inner") {
        future::plan("multisession", workers = max(1L, future::availableCores() - 1L))
        on.exit(future::plan("sequential"), add = TRUE)
        fold_apply = function(X, FUN) {
          future.apply::future_sapply(X, FUN, future.seed = TRUE)
        }
      } else {
        fold_apply = function(X, FUN) sapply(X, FUN)
      }

      ## ---------- container for optimal c per component -------------
      C_star = matrix(NA_real_, B, K_max,
        dimnames = list(names(blocks),
          paste0("PC", seq_len(K_max))))

      for (k in seq_len(K_max)) {
        lgr$info("➡️  MB-sPCA component %d / %d", k, K_max)

        ## ----- search space -----------------------------------------
        ps_k = do.call(
          paradox::ps,
          setNames(lapply(names(blocks), function(bn) {
            paradox::p_int(lower = 1L,
              upper = ncol(X_residual[[bn]]))
          }), paste0("c_", names(blocks)))
        )

        ## ----- objective (CV mean explained variance) ---------------
        cache = new.env(parent = emptyenv())
        obj_fun = bbotk::ObjectiveRFun$new(
          fun = function(xs) {
            key = paste(unlist(xs, use.names = FALSE), collapse = "_")
            if (exists(key, envir = cache, inherits = FALSE)) {
              return(list(Score = cache[[key]]))
            }

            c_vec = sqrt(unlist(xs, use.names = FALSE))

            fold_scores = fold_apply(seq_len(rs$iters), function(f) {
              Wfit = cpp_mbspca_one_lv(fold_tr[[f]], c_vec,
                max_iter = 50L, tol = 1e-4)$W

              ## project validation data
              T_val = lapply(seq_len(B), function(b) {
                fold_val[[f]][[b]] %*% as.numeric(Wfit[[b]])
              })

              ## align signs (avoid cancellation)
              for (b in 2:B) {
                s = stats::cor(T_val[[1]], T_val[[b]])
                if (is.finite(s) && s < 0) T_val[[b]] <- -T_val[[b]]
              }

              t_global = Reduce(`+`, T_val)
              num = sum(t_global^2)
              denom = Reduce(`+`, lapply(fold_val[[f]], \(M) sum(M^2)))
              if (denom < 1e-12) 0 else num / denom
            })

            score = mean(fold_scores)
            cache[[key]] = score
            list(Score = score)
          },
          domain = ps_k,
          codomain = paradox::ps(Score = paradox::p_dbl(tags = "maximize"))
        )

        inst_k = bbotk::OptimInstanceBatchSingleCrit$new(
          obj_fun, ps_k,
          terminator = bbotk::trm("evals", n_evals = private$.budget)
        )
        bbotk::opt(private$.tuner)$optimize(inst_k)

        C_star[, k] = sqrt(unlist(inst_k$result_x_domain, use.names = FALSE))
        lgr$info("    chosen c-vector: %s", paste(C_star[, k], collapse = ", "))

        ## ----- refit on full residuals ------------------------------
        fit_full = cpp_mbspca_one_lv(X_residual, C_star[, k],
          max_iter = 50L, tol = 1e-4)

        ## ----- permutation test -------------------------------------
        p_val = NA_real_
        if (private$.early_stop) {
          p_val = perm_test_component_mbspca(
            X_residual, fit_full$W, C_star[, k],
            n_perm = private$.n_perm, alpha = private$.perm_alpha
          )
          lgr$info("    permutation p-value = %.4g", p_val)

          if (p_val > private$.perm_alpha) {
            lgr$info("    🔴 early stop triggered (component not significant)")
            if (k == 1L) { # keep PC‑1 regardless
              C_star = C_star[, 1, drop = FALSE]
            } else {
              C_star = C_star[, seq_len(k - 1L), drop = FALSE]
            }
            break
          }
        }

        ## ----- deflate residuals (all folds & full) ----------------
        X_residual = private$.deflate_blocks(X_residual, fit_full$W)
        for (f in seq_len(rs$iters)) {
          fold_tr[[f]] = private$.deflate_blocks(fold_tr[[f]], fit_full$W)
          fold_val[[f]] = private$.deflate_blocks(fold_val[[f]], fit_full$W)
        }
      }

      ## ---------- store result into TuningInstance -----------------
      inst$assign_result(
        xdt = data.table::data.table(),
        y = setNames(max(inst_k$archive$data$Score), "mbspca.mean_ev"),
        learner_param_vals = list(c_matrix = C_star)
      )
      invisible(inst)
    },

    # configuration slots
    .tuner = NULL,
    .budget = NULL,
    .resampling_tpl = NULL,
    .parallel = NULL,
    .early_stop = NULL,
    .n_perm = NULL,
    .perm_alpha = NULL
  )
)
