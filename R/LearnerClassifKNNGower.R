#' k-Nearest Neighbours with Gower Distance - Classification
#'
#' @title Learner: `classif.knngower`
#' @name LearnerClassifKNNGower
#' @format R6 class inheriting from [mlr3::LearnerClassif]
#'
#' @description
#' A fast *k*-nearest neighbours classifier using **Gower distance**
#' (Rcpp/Armadillo backend). Handles numeric, unordered factors, and
#' ordered factors: numeric features are range-normalized, unordered
#' factors contribute 0/1 mismatches, and ordered factors are mapped to
#' `[0, 1]` and compared by absolute differences. Missing values are
#' skipped pairwise in the distance.
#'
#' @details
#' - **Numeric**: each column scaled by its training *range* (min-max).
#' - **Categorical**: integer-coded with `0 = NA`, `1..L = known levels`,
#'   and `-1 = unseen test level` (forces mismatch).
#' - **Ordered**: integer codes `1..L` scaled to `[0, 1]` as
#'   `(code - 1)/(L - 1)`; `NA` skipped pairwise.
#'
#' If no neighbour has sufficient comparable features, the prediction
#' falls back to the **training class priors**.
#'
#' @section Parameters (in `param_set`):
#' \describe{
#'   \item{`k`}{`integer(1)`. Number of neighbours. Default: `5`.}
#'   \item{`weights`}{`character(1)`. `"uniform"` or `"inverse"`
#'     (inverse-distance). Default: `"inverse"`.}
#'   \item{`min_feature_frac`}{`numeric(1)` in `[0,1]`. Minimum fraction of
#'     total features (numeric + categorical + ordered) that must be
#'     comparable for a neighbour to be eligible. Default: `0.2`.}
#'   \item{`na_handling`}{`character(1)`. `"pairwise"` to skip per-feature
#'     missing values (Gower), or `"fail"` to error if training features
#'     contain `NA`. Default: `"pairwise"`.}
#' }
#'
#' @section Prediction:
#' - `predict_type = "prob"` (default): class probabilities (vote shares).
#' - `predict_type = "response"`: most probable class.
#'
#' @param id `character(1)`. Learner identifier (default `"classif.knngower"`).
#' @param param_vals `list()`. Optional named list of initial hyperparameters.
#'
#' @return An object of class `LearnerClassifKNNGower`.
#'
#' @examples
#' if (requireNamespace("mlr3", quietly = TRUE)) {
#'   library(mlr3)
#'   task = tsk("iris")
#'   lrn = lrn("classif.knngower", k = 7, predict_type = "prob")
#'   lrn$train(task)
#'   p = lrn$predict(task)
#'   p$score(msr("classif.logloss"))
#' }
#'
#' @references
#' Gower, J. C. (1971). A general coefficient of similarity and some of its
#' properties. *Biometrics*, 27(4), 857-874.
#'
#' @export
LearnerClassifKNNGower = R6::R6Class("LearnerClassifKNNGower",
  inherit = mlr3::LearnerClassif,
  public = list(
    #' @description Create a new `LearnerClassifKNNGower` instance.
    initialize = function() {
      ps = paradox::ps(
        k = paradox::p_int(lower = 1L, default = 5L, tags = c("train", "predict")),
        weights = paradox::p_fct(levels = c("uniform", "inverse"),
          default = "inverse", tags = "predict"),
        min_feature_frac = paradox::p_dbl(lower = 0, upper = 1,
          default = 0.2, tags = "predict"),
        na_handling = paradox::p_fct(levels = c("pairwise", "fail"),
          default = "pairwise", tags = c("train", "predict"))
      )
      ps$set_values(k = 5L, weights = "inverse",
        min_feature_frac = 0.2, na_handling = "pairwise")

      super$initialize(
        id = "classif.knngower",
        feature_types = c("numeric", "integer", "factor", "ordered"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("twoclass", "multiclass", "missings"),
        packages = "mlr3mbspls"
      )
    }
  ),
  private = list(

    # encode numeric / categorical / ordered into three blocks suitable for C++
    .encode_blocks = function(df, num_cols, cat_cols, ord_cols, ref = NULL) {
      n = nrow(df)

      # numeric
      if (length(num_cols)) {
        Xn = as.matrix(df[, num_cols, with = FALSE])
        storage.mode(Xn) = "double"
        if (is.null(ref)) {
          r_min = suppressWarnings(apply(Xn, 2, min, na.rm = TRUE))
          r_max = suppressWarnings(apply(Xn, 2, max, na.rm = TRUE))
          rng = r_max - r_min
          rng[!is.finite(rng) | rng <= 0] = 1.0
        } else {
          rng = ref$ranges_num
        }
      } else {
        Xn = matrix(numeric(0), nrow = n, ncol = 0)
        rng = numeric(0)
      }

      # categorical (unordered) -> integer codes (0=NA, 1..L known, -1 unseen in predict)
      if (length(cat_cols)) {
        if (is.null(ref)) {
          cat_levels = lapply(cat_cols, function(cn) levels(as.factor(df[[cn]])))
        } else {
          cat_levels = ref$cat_levels
        }
        Xc = matrix(0L, nrow = n, ncol = length(cat_cols))
        for (j in seq_along(cat_cols)) {
          x = df[[cat_cols[j]]]
          lv = cat_levels[[j]]
          if (is.null(ref)) {
            code = as.integer(factor(x, levels = lv))
            code[is.na(code)] = 0L
          } else {
            m = match(as.character(x), lv)
            code = ifelse(is.na(x), 0L, ifelse(is.na(m), -1L, as.integer(m)))
          }
          Xc[, j] = code
        }
        storage.mode(Xc) = "integer"
      } else {
        Xc = matrix(integer(0), nrow = n, ncol = 0)
        cat_levels = list()
      }

      # ordered -> [0,1] via (code-1)/(L-1); unseen in predict => NA
      if (length(ord_cols)) {
        if (is.null(ref)) {
          ord_levels = lapply(ord_cols, function(cn) levels(as.ordered(df[[cn]])))
        } else {
          ord_levels = ref$ord_levels
        }
        Xo = matrix(NA_real_, nrow = n, ncol = length(ord_cols))
        for (j in seq_along(ord_cols)) {
          x = df[[ord_cols[j]]]
          lv = ord_levels[[j]]
          if (is.null(ref)) {
            code = as.integer(as.ordered(x))
          } else {
            m = match(as.character(x), lv)
            code = ifelse(is.na(x), NA_integer_, as.integer(m))
          }
          L = length(lv)
          if (L <= 1L) {
            Xo[, j] = 0
          } else {
            Xo[, j] = (as.numeric(code) - 1) / (L - 1)
          }
        }
        storage.mode(Xo) = "double"
      } else {
        Xo = matrix(numeric(0), nrow = n, ncol = 0)
        ord_levels = list()
      }

      list(
        Xnum = Xn, Xcat = Xc, Xord = Xo,
        ranges_num = as.numeric(rng),
        cat_levels = cat_levels,
        ord_levels = ord_levels
      )
    },

    .train = function(task) {
      pv = self$param_set$get_values(tags = "train")
      if (identical(pv$na_handling, "fail")) {
        x_all = task$data(cols = task$feature_names)
        if (anyNA(x_all)) stop("Training data contain missing features (na_handling = 'fail').")
      }

      df = task$data(cols = task$feature_names)
      types = task$feature_types
      num_cols = types[type %in% c("numeric", "integer"), id]
      cat_cols = types[type %in% c("factor"), id]
      ord_cols = types[type %in% c("ordered"), id]

      enc = private$.encode_blocks(df, num_cols, cat_cols, ord_cols, ref = NULL)

      y = task$truth()
      levs = levels(y)
      if (is.null(levs)) levs = sort(unique(as.character(y)))
      y_int0 = as.integer(factor(y, levels = levs)) - 1L # 0..C-1

      tab = table(factor(y, levels = levs))
      priors = as.numeric(tab) / sum(tab)

      list(
        num_cols = num_cols,
        cat_cols = cat_cols,
        ord_cols = ord_cols,
        Xnum = enc$Xnum,
        Xcat = enc$Xcat,
        Xord = enc$Xord,
        ranges_num = enc$ranges_num,
        cat_levels = enc$cat_levels,
        ord_levels = enc$ord_levels,
        y_int0 = y_int0,
        class_levels = levs,
        priors = priors
      )
    },

    .predict = function(task) {
      st = self$model
      pv = self$param_set$get_values(tags = "predict")

      df = task$data(cols = task$feature_names)
      enc_te = private$.encode_blocks(
        df,
        num_cols = st$num_cols,
        cat_cols = st$cat_cols,
        ord_cols = st$ord_cols,
        ref = list(
          ranges_num = st$ranges_num,
          cat_levels = st$cat_levels,
          ord_levels = st$ord_levels
        )
      )

      # NOTE: pass arguments POSITIONALLY (no names) to avoid "unused arguments"
      out = knn_predict_classif_gower_cpp(
        st$Xnum, st$Xcat, st$Xord,
        enc_te$Xnum, enc_te$Xcat, enc_te$Xord,
        st$ranges_num, st$y_int0,
        as.integer(pv$k),
        pv$weights,
        as.numeric(pv$min_feature_frac),
        st$priors
      )

      prob = as.matrix(out$prob)
      colnames(prob) = st$class_levels

      response = factor(st$class_levels[max.col(prob, ties.method = "first")],
        levels = st$class_levels)

      if (self$predict_type == "prob") {
        mlr3::PredictionClassif$new(task = task, response = response, prob = prob)
      } else {
        mlr3::PredictionClassif$new(task = task, response = response)
      }
    }
  )
)
