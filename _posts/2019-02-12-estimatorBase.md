---
layout: post
title:  "estimator"
categories: estimator
tags: estimator tensorflow
author: admin
---

* content
{:toc}

### Estimator  
![estimator](../img/estimator.png "estimator")  
estimator 是tensorflow的高级API，对数据和模型提供了编程接口，方便了程序的开发
可以看到estimator 依赖Mid-level API  
> 1.Layers:构建网络  
> 2.Datesets :  数据处理  
> 3.Metrics:评价函数  

estimator 类  
```
class Estimator(object):
  """Estimator class to train and evaluate TensorFlow models.

  The `Estimator` object wraps a model which is specified by a `model_fn`,
  which, given inputs and a number of other parameters, returns the ops
  necessary to perform training, evaluation, or predictions.

  All outputs (checkpoints, event files, etc.) are written to `model_dir`, or a
  subdirectory thereof. If `model_dir` is not set, a temporary directory is
  used.

  The `config` argument can be passed `tf.estimator.RunConfig` object containing
  information about the execution environment. It is passed on to the
  `model_fn`, if the `model_fn` has a parameter named "config" (and input
  functions in the same manner). If the `config` parameter is not passed, it is
  instantiated by the `Estimator`. Not passing config means that defaults useful
  for local execution are used. `Estimator` makes config available to the model
  (for instance, to allow specialization based on the number of workers
  available), and also uses some of its fields to control internals, especially
  regarding checkpointing.

  The `params` argument contains hyperparameters. It is passed to the
  `model_fn`, if the `model_fn` has a parameter named "params", and to the input
  functions in the same manner. `Estimator` only passes params along, it does
  not inspect it. The structure of `params` is therefore entirely up to the
  developer.

  None of `Estimator`'s methods can be overridden in subclasses (its
  constructor enforces this). Subclasses should use `model_fn` to configure
  the base class, and may add methods implementing specialized functionality.

  @compatibility(eager)
  Calling methods of `Estimator` will work while eager execution is enabled.
  However, the `model_fn` and `input_fn` is not executed eagerly, `Estimator`
  will switch to graph model before calling all user-provided functions (incl.
  hooks), so their code has to be compatible with graph mode execution. Note
  that `input_fn` code using `tf.data` generally works in both graph and eager
  modes.
  @end_compatibility
  """

  def __init__(self, model_fn, model_dir=None, config=None, params=None,
               warm_start_from=None):
    """Constructs an `Estimator` instance.

    See [estimators](https://tensorflow.org/guide/estimators) for more
    information.

    To warm-start an `Estimator`:

    ```python
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
        hidden_units=[1024, 512, 256],
        warm_start_from="/path/to/checkpoint/dir")
    ```

    For more details on warm-start configuration, see
    `tf.estimator.WarmStartSettings`.

    Args:
      model_fn: Model function. Follows the signature:

        * Args:

          * `features`: This is the first item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `tf.Tensor` or `dict` of same.
          * `labels`: This is the second item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `tf.Tensor` or `dict` of same (for multi-head models).
                 If mode is @{tf.estimator.ModeKeys.PREDICT}, `labels=None` will
                 be passed. If the `model_fn`'s signature does not accept
                 `mode`, the `model_fn` must still be able to handle
                 `labels=None`.
          * `mode`: Optional. Specifies if this training, evaluation or
                 prediction. See `tf.estimator.ModeKeys`.
          * `params`: Optional `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config`: Optional `estimator.RunConfig` object. Will receive what
                 is passed to Estimator as its `config` parameter, or a default
                 value. Allows setting up things in your `model_fn` based on
                 configuration such as `num_ps_replicas`, or `model_dir`.

        * Returns:
          `tf.estimator.EstimatorSpec`

      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator to
        continue training a previously saved model. If `PathLike` object, the
        path will be resolved. If `None`, the model_dir in `config` will be used
        if set. If both are set, they must be same. If both are `None`, a
        temporary directory will be used.
      config: `estimator.RunConfig` configuration object.
      params: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.
      warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                       warm-start from, or a `tf.estimator.WarmStartSettings`
                       object to fully configure warm-starting.  If the string
                       filepath is provided instead of a
                       `tf.estimator.WarmStartSettings`, then all variables are
                       warm-started, and it is assumed that vocabularies
                       and `tf.Tensor` names are unchanged.

    Raises:
      ValueError: parameters of `model_fn` don't match `params`.
      ValueError: if this is called via a subclass and if that class overrides
        a member of `Estimator`.
    """
```
> <font color="blue">可以看出model_fn参数是一个函数：def model_fn(features,labels,mode,params,config)</font>  
> <font color="blue">config:是一个对象由estimator.RunConfig创建（ 主要是训练是的一些参数配置）</font>  
> <font color="blue">params：一些其它的参数，形式为dict(),比如batch_size大小等</font>  
> <font color="blue">warm_start_from：指定checkpoint路径，会导入该checkpoint开始训练</font>  
> 
<font color="red">由这几个参数，详细说明estimator如何构建模型、进行数据处理以及训练的</font>

> params：指定额外的参数，形式为字典，比如：

```
params=dict()
params["train_batch"]=128
params["eval_batch"]=64
```

><font color="blue">config:是一个对象，由estimator.RunConfig创建，改参数指定了模型训练的基本配置，比如模型保存 路径，多久保存一次模型，多久输出一次日志等等。</font>

```
 run_config = tf.estimator.RunConfig(model_dir=None, 
        tf_random_seed=None, 
        save_summary_steps=100, 
        save_checkpoints_steps=_USE_DEFAULT, 
        save_checkpoints_secs=_USE_DEFAULT, 
        session_config=None, 
        keep_checkpoint_max=5, 
        keep_checkpoint_every_n_hours=10000, 
        log_step_count_steps=100, 
        train_distribute=None, 
        device_fn=None, 
        protocol=None, 
        eval_distribute=None, 
        experimental_distribute=None)
 这里是全部的参数，大多数时间，我们只需要指定部分参数，其余默认即可。比如
 run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=100,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=20,
        log_step_count_steps=25)
参数的意思都比较容易明白，其中train_distribute参数是指定分布式策略的，如果需要模型多GPU并行或者多机器执行，需要用这个参数，此处可以忽略。
session_config指定了会话的参数一般用来指定使用GPU的策略，比如是否按需分配显存还是独占式，比如
		sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
在tensorflow 中创建会话经常会用到，在不使用estimator的情况下一般是：
         sess = tf.Session(config=sess_config)
****相信此时你有疑问：模型在哪里创建，数据如何输入，请保持好奇心！****
```

> <font color="blue">参数model_fn是一个函数</font>

```
def model_fn(features,labels,mode,params):
  #### your code
根据estimator对model_fn参数的说明可以知道，该函数是一个重要的接口，它负责接受数据和模型的训练，验证测试等

参数features：该参数接收函数input_fn函数的输出，由input_fn函数提供模型所需的数据；features可以是一个tensor或者是字典，多数情况下是字典，因为模型一般需要多个输入。

参数labels:该参数接收函数input_fn函数的输出，形式也是一个tensor或者字典,多数情况下标签的值被封装在第一参数中，这种个情况下input可以不用给labels返回值，或者返回None

mode 参数，取值为tf.estimator.ModeKeys，当estimator运行模式指定后，mode就固定了。
调用estimator.train()对应 mode为tf.estimator.ModeKeys.TRAIN，
调用estimator.evaluate() mode为tf.estimator.ModeKeys.EVAL，
调用estimator.predict()为tf.estimator.ModeKeys.PREDICT
```

<font color="blue">可以看出模型需要在model_fn函数中调用我们的模型，通过mode 参数控制训练、验证、预测；数据由features和labels参数接受数据输入函数input_fn的结果；</font>

input_fn如何为模型提供数据的，什么时候提供数据的？实际上只有当训练的时候，才会调用input_fn 函数进行数据准备，因此使用estimator API需要准备两个部件：model_fn和input_fn,下面以BERT原始代码说明。

### BERT Example

```
（这里已经修改使用GPU训练而非TPU)
def main(_):
  ------------------------------
            Other Code
  ------------------------------
  
 ###extimator config参数的初始化
  run_config = tf.estimator.RunConfig(
            model_dir=FLAGS.output_dir,
            save_summary_steps=100,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=20,
            log_step_count_steps=25)
 ###extimator config参数的初始化
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  
  
  ### extimator参数model_fn函数的构建
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      use_multi_gpu=FLAGS.use_multi_gpu
  )
 ### extimator参数model_fn函数的构建
 
 ### params参数的构建
  params=dict()
  params["batch_size"]=FLAGS.train_batch_size
 ### params参数的构建
 
 ### 构建estimator
  estimator=tf.estimator.Estimator(
        model_fn=model_fn, 
        model_dir=FLAGS.output_dir, 
        config=run_config,
        params=params,
        warm_start_from=None)
  ### 构建estimator
  
  基本到此estimator大致结构已经清晰，但是具体的构建一个model_fn和input_fn还是需要一点工总量的，首先说明input_fn的构建；input_fn函数是最后一步训练/验证/预测的时候用的，在这之前需要做一些必要的数据处理。
```

###  input__fn 函数构建

> input_fn函数返回给模型的是分完batch的结构化数据，estimator中是字典封装好的tensor

> 过程(BERT为例)：首先读取原始数据文件得到train_examples/eval_examples---------------> 原始数据进行填充，查表等操作转化为字典形式的结构化数据以TFRecord文件存储（可以存储为其它形式 ---------->之后解析数据，分batch以Dataset形式送给模型

```
train_examples = processor.get_train_examples(FLAGS.data_dir)//加载原始数据
file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)    //原始数据进行填充，查表等操作转化为字典形式的结构化数据以TFRecord文件存储
train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True) //构建input_fn函数
     
//函数file_based_convert_examples_to_features的定义
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,max_seq_length, tokenizer) //进行必要的查表填充对齐等操作

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
  //函数file_based_convert_examples_to_features的定义
  
//file_based_input_fn_builder函数定义
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn
  //file_based_input_fn_builder函数定义
```

### model_fn 构建

model_fn函数主要负责模型的创建以及训练、验证、预测等一系列的操作

```
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, use_multi_gpu):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    //创建模型
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          use_multi_gpu)
      output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metric_ops=metric_fn(per_example_loss,label_ids,logits,is_real_example)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metric_ops)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities})
    return output_spec

  return model_fn
```

注意model_fn_builder和file_based_input_fn_builder函数返回的分别是内置的model_fn和input_fn函数

以上就是使用estimator的简单说明，关于hook和tf的分布式累积更新有时间再更新