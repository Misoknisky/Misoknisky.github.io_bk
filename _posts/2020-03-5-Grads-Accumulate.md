---
layout: post
title:  "accumulate grads"
categories: grads gpus
tags: grads tensorflow
author: admin
---

* content
{:toc}

tensorflow 是目前比较流行的深度学习框架，限制与我们的设备问题，在使用大数据训练模型的时候发现需要用到梯度累积技术和GPU并行计算的技术，GPU并行的方式在tensorflow的高阶API estimator中有提供，但是对于初学者来说可能并不了解Estimator API，这里给出一般情况下（不实用estimator)如何进行并行和梯度累积，这里介绍的并行是数据并行，相关概念可以自行百度
### Tensorflow 梯度计算
梯度函数minimize实际上调用了两个函数compute_gradients、apply_gradients，前者负责计算梯度，后者负责进行梯度更新，为了开发者可以对梯度进行一定的操作（比如裁剪）tensorflow提供了minimize函数的同时也提供了compute_gradients、apply_gradients函数，方便我们灵活使用。
 以下是minimize 函数的源码说明，供参考  

```
def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in
        the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.
```

### compute_gradients()

一下是compute_gradients函数的源码说明，供参考

```
def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking
        no arguments which returns the value to minimize. When eager execution
        is enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
```
可以发现compute_gradients函数返回的是梯度对[(gradient1, variable1),(gradient2, variable2),...]，gradient 表示的梯度值，variable表示的是变量，也就是该变量variable的梯度是gradient。除此之外梯度可能出现没有梯度的情况即梯度为 None，实际情况下在模型中计算出来的梯度是什么样子的呢，以下给出一个样例(gradient, variable)：

```
IndexedSlices(indices=Tensor("model/gradients/concat_1:0", shape=(?,), dtype=int32, device=/device:GPU:3), values=Tensor("model/gradients/concat:0", shape=(?, 300), dtype=float32, device=/device:GPU:3), dense_shape=Tensor("model/gradients/model/word_embedding/embedding_lookup_grad/ToInt32:0", shape=(2,), dtype=int32, device=/device:CPU:0)) <tf.Variable 'model/word_embedding/word_embeddings:0' shape=(45438, 300) dtype=float32_ref>
Tensor("model/gradients/model/passage_encoding/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(450, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/passage_encoding/bidirectional_rnn/fw/lstm_cell/kernel:0' shape=(450, 600) dtype=float32_ref>
Tensor("model/gradients/model/passage_encoding/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/passage_encoding/bidirectional_rnn/fw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/passage_encoding/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(450, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/passage_encoding/bidirectional_rnn/bw/lstm_cell/kernel:0' shape=(450, 600) dtype=float32_ref>
Tensor("model/gradients/model/passage_encoding/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/passage_encoding/bidirectional_rnn/bw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/question_encoding/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(450, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/question_encoding/bidirectional_rnn/fw/lstm_cell/kernel:0' shape=(450, 600) dtype=float32_ref>
Tensor("model/gradients/model/question_encoding/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/question_encoding/bidirectional_rnn/fw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/question_encoding/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(450, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/question_encoding/bidirectional_rnn/bw/lstm_cell/kernel:0' shape=(450, 600) dtype=float32_ref>
Tensor("model/gradients/model/question_encoding/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/question_encoding/bidirectional_rnn/bw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/fusion/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(1350, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/fusion/bidirectional_rnn/fw/lstm_cell/kernel:0' shape=(1350, 600) dtype=float32_ref>
Tensor("model/gradients/model/fusion/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/fusion/bidirectional_rnn/fw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/fusion/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(1350, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/fusion/bidirectional_rnn/bw/lstm_cell/kernel:0' shape=(1350, 600) dtype=float32_ref>
Tensor("model/gradients/model/fusion/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/fusion/bidirectional_rnn/bw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/fusion/gated_layer/dense/MatMul_grad/tuple/control_dependency_1:0", shape=(300, 300), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/fusion/gated_layer/dense/W:0' shape=(300, 300) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/self_attention/MatMul_grad/tuple/control_dependency_1:0", shape=(300, 300), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/self_attention/W:0' shape=(300, 300) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/fw/fw/while/gru_cell/MatMul/Enter_grad/b_acc_3:0", shape=(750, 300), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/fw/gru_cell/gates/kernel:0' shape=(750, 300) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/fw/fw/while/gru_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(300,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/fw/gru_cell/gates/bias:0' shape=(300,) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/fw/fw/while/gru_cell/MatMul_1/Enter_grad/b_acc_3:0", shape=(750, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/fw/gru_cell/candidate/kernel:0' shape=(750, 150) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/fw/fw/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_3:0", shape=(150,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/fw/gru_cell/candidate/bias:0' shape=(150,) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/bw/bw/while/gru_cell/MatMul/Enter_grad/b_acc_3:0", shape=(750, 300), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/bw/gru_cell/gates/kernel:0' shape=(750, 300) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/bw/bw/while/gru_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(300,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/bw/gru_cell/gates/bias:0' shape=(300,) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/bw/bw/while/gru_cell/MatMul_1/Enter_grad/b_acc_3:0", shape=(750, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/bw/gru_cell/candidate/kernel:0' shape=(750, 150) dtype=float32_ref>
Tensor("model/gradients/model/para_attention/bidirectional_rnn/bw/bw/while/gru_cell/BiasAdd_1/Enter_grad/b_acc_3:0", shape=(150,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/para_attention/bidirectional_rnn/bw/gru_cell/candidate/bias:0' shape=(150,) dtype=float32_ref>
Tensor("model/gradients/model/document_attention/self_attention/MatMul_grad/tuple/control_dependency_1:0", shape=(300, 300), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/document_attention/self_attention/W:0' shape=(300, 300) dtype=float32_ref>
Tensor("model/gradients/model/document_attention/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(750, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/document_attention/bidirectional_rnn/fw/lstm_cell/kernel:0' shape=(750, 600) dtype=float32_ref>
Tensor("model/gradients/model/document_attention/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/document_attention/bidirectional_rnn/fw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/document_attention/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3:0", shape=(750, 600), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/document_attention/bidirectional_rnn/bw/lstm_cell/kernel:0' shape=(750, 600) dtype=float32_ref>
Tensor("model/gradients/model/document_attention/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3:0", shape=(600,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/document_attention/bidirectional_rnn/bw/lstm_cell/bias:0' shape=(600,) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/attend_pooling/ExpandDims_grad/Reshape:0", shape=(1, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/random_attn_vector:0' shape=(1, 150) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/attend_pooling/fully_connected/Tensordot/transpose_1_grad/transpose:0", shape=(300, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/attend_pooling/fully_connected/weights:0' shape=(300, 150) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/attend_pooling/fully_connected_1/Tensordot/transpose_1_grad/transpose:0", shape=(150, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/attend_pooling/fully_connected_1/weights:0' shape=(150, 150) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/attend_pooling/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1:0", shape=(150,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/attend_pooling/fully_connected_1/biases:0' shape=(150,) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/attend_pooling/fully_connected_2/Tensordot/transpose_1_grad/transpose:0", shape=(150, 1), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/attend_pooling/fully_connected_2/weights:0' shape=(150, 1) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/attend_pooling/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1:0", shape=(1,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/attend_pooling/fully_connected_2/biases:0' shape=(1,) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/fully_connected/MatMul_grad/tuple/control_dependency_1:0", shape=(300, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/fully_connected/weights:0' shape=(300, 150) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/fully_connected/BiasAdd_grad/tuple/control_dependency_1:0", shape=(150,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/fully_connected/biases:0' shape=(150,) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/fw/fully_connected/Tensordot/transpose_1_grad/transpose:0", shape=(300, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/fw/fully_connected/weights:0' shape=(300, 150) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/fw/fully_connected/BiasAdd_grad/tuple/control_dependency_1:0", shape=(150,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/fw/fully_connected/biases:0' shape=(150,) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/fw/while/PointerNetLSTMCell/fully_connected/MatMul/Enter_grad/b_acc_3:0", shape=(150, 150), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/fw/PointerNetLSTMCell/fully_connected/weights:0' shape=(150, 150) dtype=float32_ref>
Tensor("model/gradients/model/pn_decoder/fw/while/PointerNetLSTMCell/fully_connected/BiasAdd/Enter_grad/b_acc_3:0", shape=(150,), dtype=float32, device=/device:GPU:3) <tf.Variable 'model/pn_decoder/fw/PointerNetLSTMCell/fully_connected/biases:0' shape=(150,) dtype=float32_ref>
```

可以发现gradient多数情况下是一个Tensor,除了Tensor外还有一个IndexedSlices类型；variable总是一个tf.Variable类型；正常情况下计算一个batch的梯度更新一次，现在我们不想马上更新梯度，希望累积到一定步数更新一次；正常的思路就是将所有累积的梯度使用一个变量进行保存，然后更新，这里就需要将梯度计算和更新分开来做。 tensorflow在2.0之前是静态图，静态图将累积的梯度回传的方式就是通过feed机制，因此我们需要的梯度储存器是一个placeholder占位符如何创建？variable表死的是计算图的节点，当图确定后计算图的节点是不变的，它表示的是节点的名称，我们之要将累计的梯度值和节点一一对应上即可，所以对Variable 不需要累积，只需要把梯度值进行累计即可；对于tensor类型的gradient：

```
grads_holede=[]
grads_holede.append((tf.placeholder(dtype=g.dtype, shape=g.get_shape()), v))
```
按以上方式创建梯度接收器，梯度的形式必须和原来保持一致，因此这里封装的是(gradient,Variable) 的元组。之后只需将求得的梯度累积求和feed给梯度接收器进行更新即可。

对于gradient为IndexedSlices类型比较麻烦一些，IndexedSlices类型是一个稀疏矩阵，是因为查embedding表操作产生的，可以看出所有的梯度中只维护了一个IndexedSlices类型，而实际上程序中有两个查表操作（我的程序是这么写的），研究发现这个IndexedSlices变量拼接了所有产生的稀疏矩阵；IndexedSlices 有三个值indices、values、dense_shape，其中indices表示需要更新的embedding表对应的行，values和indices一一对应代表查表获取的相应的embedding 表示，dense_shape表示的是整个embedding表的大小，和IndexedSlices对应的Variable就是embedding节点名称。可以发现IndexedSlices 里的indices、values、dense_shape都是tensor因此都需要创建对应的接受器，然后再封装成IndexedSlices，将封装后的IndexedSlices作为gradient与对应的Variable一起构成梯度元组。具体的如下：
```
IndexedSlices_index = tf.placeholder(dtype=tf.int32,shape=g.indices.shape)
IndexedSlices_value = tf.placeholder(dtype=g.values.dtype,shape=g.values.shape)
IndexedSlices_Dense_shape = tf.placeholder(dtype=g.dense_shape.dtype, shape=g.dense_shape.shape)
grade_IndexedSlices=tf.IndexedSlices(self.IndexedSlices_value,
                                                     self.IndexedSlices_index,
                                                     dense_shape=self.IndexedSlices_Dense_shape)
grads_holder.append((grade_IndexedSlices,v))
```



完整代码如下所示(AccumulateSteps类定义)

```
class AccumulateSteps(object):
    def __init__(self,grads_vars,accumulate_step=2):
        """
            grads_vars:[(g1,v1),(g2,v2)]
        """
        assert accumulate_step >0
        self.grads_holder=[]
        self.grads_accumulator=collections.OrderedDict()
        self.local_step=0
        self.accumulate_step=accumulate_step
        self.IndexedSlices_index=None
        self.IndexedSlices_value=None
        self.IndexedSlices_Dense_shape=None
        for (g, v) in grads_vars:
            if g is None: continue
            if isinstance(g, tf.IndexedSlices):
                self.IndexedSlices_index = tf.placeholder(dtype=g.dtype,shape=g.indices.shape)
                self.IndexedSlices_value = tf.placeholder(dtype=g.values.dtype,shape=g.values.shape)
                self.IndexedSlices_Dense_shape = tf.placeholder(dtype=g.dense_shape.dtype, shape=g.dense_shape.shape)
                grade_IndexedSlices=tf.IndexedSlices(self.IndexedSlices_value,
                                                     self.IndexedSlices_index,
                                                     dense_shape=self.IndexedSlices_Dense_shape)
                self.grads_holder.append((grade_IndexedSlices,v))
            else:
                self.grads_holder.append((tf.placeholder(dtype=g.dtype, shape=g.get_shape()), v))
    def set_local_step(self,step):
        self.local_step =step
    def _generate_grads_dict(self):
        feed_dict={}
        feed_dict.update({self.IndexedSlices_index:self.grads_accumulator["indices"],
                            self.IndexedSlices_value:self.grads_accumulator["values"],
                            self.IndexedSlices_Dense_shape:self.grads_accumulator["dense_shape"]})
        for holder_index,placeholder in enumerate(self.grads_holder):
            if holder_index <=0:continue
            feed_dict.update({placeholder[0]:self.grads_accumulator[holder_index]})
        self.grads_accumulator.clear()
        self.local_step=0
        return feed_dict
    def add_grads(self,grad_vars,right_grads=False):
        if right_grads:
            return self._generate_grads_dict()
        self.local_step +=1
        assert len(grad_vars) == len(self.grads_holder)
        for g_uid,(v,g) in enumerate(grad_vars):
            if isinstance(g,IndexedSlicesValue):
                if "indices" in self.grads_accumulator:
                    self.grads_accumulator["indices"] = np.concatenate((self.grads_accumulator["indices"],g.indices),axis=0)
                else:
                    self.grads_accumulator["indices"] = g.indices
                if "values" in self.grads_accumulator:
                    self.grads_accumulator["values"] = np.concatenate((self.grads_accumulator["values"],g.values),axis=0)
                else:
                    self.grads_accumulator["values"] = g.values
                self.grads_accumulator["dense_shape"] = g.dense_shape
            else:
                if g_uid in self.grads_accumulator:
                    self.grads_accumulator[g_uid]=sum([self.grads_accumulator[g_uid],g])
                else:
                    self.grads_accumulator[g_uid]=g
        
        if self.local_step == self.accumulate_step:
            
            return self._generate_grads_dict()
        else:
            return None
```

average_gradients函数定义如下  ，多个GPU并行后收集每个device的梯度进行平均，这里采用的是数据并行，每个device共享一个计算图，具体可查看main函数

```
def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      values, new_index_positions,
      tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)
def average_gradients(tower_grads):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grad_name, grad_value = grad_and_vars[0]
        if grad_name is None:
            # no gradient for this variable, skip it
            average_grads.append((grad_name, grad_value))
            continue

        if isinstance(grad_name, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #  a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=grad_name.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                   expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over 
                   grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))
    
    return average_grads
```



对应的main.py函数

```
logger = logging.getLogger("brc")
logger.info('Load data_set and vocab...')
with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
     vocab = pickle.load(fin)
brc_data = BRCDataset(args.max_p_num, 
                          args.max_p_len, 
                          args.max_q_len,
                          args.gpus,
                          args.batch_size,
                          args.train_files, 
                          args.dev_files)
logger.info('Converting text into ids...')
brc_data.convert_to_ids(vocab)
opt=create_train_op(args.optim,args.learning_rate)
tower_grads,models= [],[]
train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
global_step = tf.get_variable(
          'global_step', [],
           dtype=tf.int32,
           initializer=tf.constant_initializer(0), trainable=False)
for k,gpu_num in enumerate(args.gpus):
    resuse_flag=True if k > 0 else False
    with tf.device('/gpu:%s' % gpu_num):
         with tf.variable_scope('model', reuse=resuse_flag):
              model=Model(vocab,args)
              models.append(model)
              loss=model.loss
              grads=opt.compute_gradients(loss)
              tower_grads.append(grads)
              train_perplexity += loss
ave_grads=average_gradients(tower_grads)
train_perplexity = train_perplexity / len(args.gpus)
accumulator=AccumulateSteps(grads_vars=ave_grads,accumulate_step=args.accumulate_n)
train_op = opt.apply_gradients(accumulator.grads_holder, global_step=global_step)
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
pad_id=vocab.get_id(vocab.pad_token)
max_rouge_l=-1
for epoch in range(1,args.epochs + 1):
    epoch_sample_num, epoch_total_loss = 0, 0
    real_batch=args.batch_size * len(args.gpus)
    train_batches = brc_data.gen_mini_batches('train',real_batch, pad_id, shuffle=True)
    for batch_num,batch_data in enumerate(train_batches,1):
        passage_num=len(batch_data['passage_token_ids']) //real_batch
        feed_dict=dict()
        for k in range(len(args.gpus)):
            start =k * args.batch_size
            end = (k+1) * args.batch_size
            feed_dict.update({....})
        grads_values,loss= sess.run([ave_grads,train_perplexity],feed_dict)
        grads_feed=accumulator.add_grads(grads_values)
        .......
        if grads_feed is not None:
           _=sess.run([train_op],feed_dict=grads_feed)
           global_step = tf.train.get_global_step().eval(session=sess)
    if epoch_sample_num % (args.batch_size * len(args.gpus)*args.accumulate_n) !=0:
       grads_feed=accumulator.add_grads(grad_vars=None,right_grads=True)
       _=sess.run([train_op],feed_dict=grads_feed)
       global_step = tf.train.get_global_step().eval(session=sess)
       .......
       
```



整个框架结构如上所示，详细的大家可以自行研究，如有更好的建议可以联系我，模型的具体代码由于实验室的规则不能开放给出实验结果

### 结果

实验训练数据为1000条，验证集100条，训练集较少模型有些许波动（机器阅读理解模型，数据集为Dureader）

batch-size 为1 累积梯度为8：Rouge-L的值41+ 、bleu4:40+

batch-size 为8 没有累计梯度：Rouge-L的值40+、bleu4:40+

batch-size 为8 GPUS为4：Rouge-L的值42+ 、bleu4:41+

batch-size 为32 GPUS为1：Rouge-L的值42+ 、bleu4:41+

其余结果就不给出了
