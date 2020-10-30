import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2, convert_variables_to_constants_v2_as_graph
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util, dtypes
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from detection.model import build_compiled_model
from detection.datasets.dataset import build_data
from detection.configs import cfg

import argparse
import os
import cv2
import numpy as np


def get_model():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--restart", default=0, type=int)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)
    
    latest = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
    model = build_compiled_model(cfg, for_export=True)
    model.load_weights(latest)

    return model, cfg
    # train_data, val_data = build_data(cfg)

    # model.fit_generator(
    #     train_data,
    #     steps_per_epoch=train_data.steps_per_epoch,
    #     validation_data=val_data,
    #     # use_multiprocessing=True,
    #     workers=6,
    #     epochs=cfg.EPOCH,
    #     callbacks=[build_checkpoint_callback(cfg)]
    # )


def for_tf2(model):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # for l in full_model.layers:
    #     print(l)
    # real_model.summary()
    # Get frozen ConcreteFunction
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(full_model)
    # print(graph_def)
    print('input: ', graph_def.node[0].op)
    print('output: ', graph_def.node[-1].op)
    for node in graph_def.node:
        # print('#'*50)
        # print('name: ' ,node.name)
        # print(node.op)
        # print([f for f in node.attr.keys()])
        if node.op == 'FusedBatchNormV3':
            del node.attr["exponential_avg_factor"]
            # print([f for f in node.attr.keys()])
            # node.attr["exponential_avg_factor"].CopyFrom(
                # attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto([1], dtypes.float32)))
    # print(frozen_func.graph._collections)
    # for f in frozen_func.graph.node:
    #     print(f)

    # input_tensors = [
    #     tensor for tensor in frozen_func.inputs
    #     if tensor.dtype != tf.resource
    # ]
    
    # output_tensors = frozen_func.outputs

    # graph_def = run_graph_optimizations(
    #     graph_def,
    #     input_tensors,
    #     output_tensors,
    #     config=get_grappler_config(["constfold", "function"]),
    #     graph=frozen_func.graph)
    
    return graph_def


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def main():

    # Create, compile and train model...
    model, cfg = get_model()
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'export')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'export'))
    model.save(os.path.join(cfg.OUTPUT_DIR, 'export'))
    print('model saved')
    # loaded = tf.keras.models.load_model(os.path.join(cfg.OUTPUT_DIR, 'export'))
    # loaded.summary()
    loaded = tf.saved_model.load(os.path.join(cfg.OUTPUT_DIR, 'export'))
    # returns = loaded([cv2.resize(cv2.imread('./tt.jpg'), (128, 256)).astype(np.float32) / 255])
    # print(returns)
    # print(loaded.layers)
    print(list(loaded.signatures.keys()))

    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)

    # print(model.inputs)
    # print(model.outputs)
    # frozen_graph = for_tf2(model)
    # tf.io.write_graph(graph_or_graph_def=frozen_graph,
    #                 logdir=cfg.OUTPUT_DIR,
    #                 name="my_model.pb",
    #                 as_text=False)
    # frozen_graph = freeze_session(K.get_session(),
    #                             output_names=[out.op.name for out in model.outputs])
    # tf.train.write_graph(frozen_graph, cfg.OUTPUT_DIR, "my_model.pb", as_text=False)


if __name__ == '__main__':
    main()
