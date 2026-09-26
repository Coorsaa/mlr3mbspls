# Preprocessing step that records the rows it is trained on.
PipeOpRecordTrainRows = R6::R6Class("PipeOpRecordTrainRows",
  inherit = mlr3pipelines::PipeOpTaskPreproc,
  public = list(
    record = NULL,
    initialize = function(id = "record_rows", record = new.env(parent = emptyenv())) {
      self$record = record
      self$record$train = list()
      super$initialize(id = id)
    }
  ),
  private = list(
    .train_task = function(task) {
      self$record$train = c(self$record$train, list(sort(task$row_ids)))
      self$state = list()
      task
    },
    .predict_task = function(task) task
  )
)
