"""
Remove Q/DQ nodes for excluded layers from ONNX model
This ensures excluded layers truly run in FP16 in TensorRT
"""

import onnx
from onnx import helper, numpy_helper
import argparse
import sys


def should_exclude_node(node_name, exclude_layers):
    """Check if a node belongs to an excluded layer"""
    if not exclude_layers:
        return False
    for excluded in exclude_layers:
        if excluded in node_name:
            return True
    return False


def find_node_by_output(graph, output_name):
    """Find node that produces given output"""
    for node in graph.node:
        if output_name in node.output:
            return node
    return None


def find_node_by_input(graph, input_name):
    """Find node that consumes given input"""
    for node in graph.node:
        if input_name in node.input:
            return node
    return None


def remove_qdq_for_excluded_layers(onnx_path, output_path, exclude_layers):
    """
    Remove QuantizeLinear and DequantizeLinear nodes for excluded layers
    by connecting the nodes before Q to the nodes after DQ
    """
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)

    if not exclude_layers:
        print("No layers to exclude, copying model as-is")
        onnx.save(model, output_path)
        return

    print(f"Excluding layers: {exclude_layers}")

    # Collect Q/DQ pairs to remove
    q_nodes_to_remove = []
    dq_nodes_to_remove = []

    for node in model.graph.node:
        if should_exclude_node(node.name, exclude_layers):
            if node.op_type == 'QuantizeLinear':
                q_nodes_to_remove.append(node)
                print(f"  Found Q node to remove: {node.name}")
            elif node.op_type == 'DequantizeLinear':
                dq_nodes_to_remove.append(node)
                print(f"  Found DQ node to remove: {node.name}")

    print(f"\nTotal Q nodes to remove: {len(q_nodes_to_remove)}")
    print(f"Total DQ nodes to remove: {len(dq_nodes_to_remove)}")

    if not q_nodes_to_remove and not dq_nodes_to_remove:
        print("No Q/DQ nodes found for excluded layers")
        onnx.save(model, output_path)
        return

    # Build a mapping: DQ output -> Q input (to reconnect the graph)
    # For each DQ node, we need to find its corresponding Q node
    reconnection_map = {}  # old_input (from DQ output) -> new_input (from Q input)

    for dq_node in dq_nodes_to_remove:
        dq_input = dq_node.input[0]  # Input to DQ node
        dq_output = dq_node.output[0]  # Output from DQ node

        # Find the corresponding Q node that feeds into this DQ
        # Usually Q.output[0] == DQ.input[0] in a Q-DQ pair
        for q_node in q_nodes_to_remove:
            if q_node.output[0] == dq_input:
                q_input = q_node.input[0]  # Original input before quantization
                reconnection_map[dq_output] = q_input
                print(f"  Will reconnect: {dq_output} -> {q_input}")
                break

    # Also handle weight quantizers where there might be only DQ nodes
    for q_node in q_nodes_to_remove:
        if q_node not in [n for n in q_nodes_to_remove if n.output[0] in [dq.input[0] for dq in dq_nodes_to_remove]]:
            # This Q node doesn't have a corresponding DQ in our list
            # It's likely a weight quantizer
            q_output = q_node.output[0]
            q_input = q_node.input[0]
            # Weight quantizers usually feed directly into Conv/MatMul
            # The weight is already quantized, so we keep it as is
            pass

    # Update connections in all nodes
    nodes_to_remove = q_nodes_to_remove + dq_nodes_to_remove

    for node in model.graph.node:
        if node not in nodes_to_remove:
            # Update inputs to reconnect the graph
            for i, input_name in enumerate(node.input):
                if input_name in reconnection_map:
                    old_input = node.input[i]
                    node.input[i] = reconnection_map[input_name]
                    print(f"  Reconnected: {node.name} input {old_input} -> {reconnection_map[input_name]}")

    # Create new graph without excluded nodes
    new_nodes = [n for n in model.graph.node if n not in nodes_to_remove]

    # Collect initializers to remove (those only used by removed nodes)
    all_removed_inputs = set()
    for node in nodes_to_remove:
        for input_name in node.input[2:]:  # Skip first two inputs (data, scale)
            all_removed_inputs.add(input_name)

    # Check if any initializers are still used by remaining nodes
    still_used = set()
    for node in new_nodes:
        for input_name in node.input:
            still_used.add(input_name)

    initializers_to_remove = all_removed_inputs - still_used

    # Remove unused initializers
    new_initializers = [i for i in model.graph.initializer
                        if i.name not in initializers_to_remove]

    # Build new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=new_initializers,
        value_info=model.graph.value_info
    )

    # Create new model
    new_model = helper.make_model(new_graph,
                                   opset_imports=model.opset_import,
                                   producer_name=model.producer_name)

    # Check and save
    try:
        onnx.checker.check_model(new_model)
        print("ONNX model validation passed")
    except Exception as e:
        print(f"Warning: ONNX validation error: {e}")
        print("Attempting to save anyway...")

    onnx.save(new_model, output_path)

    print(f"\nSaved modified ONNX to: {output_path}")
    print(f"Removed {len(nodes_to_remove)} Q/DQ nodes")
    print(f"Removed {len(model.graph.initializer) - len(new_initializers)} unused initializers")


def main():
    parser = argparse.ArgumentParser(description='Remove Q/DQ nodes for excluded layers')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Input ONNX model path')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output ONNX model path')
    parser.add_argument('--exclude_layers', type=str, default='',
                        help='Comma-separated list of layer names to exclude')

    args = parser.parse_args()

    exclude_layers = [l.strip() for l in args.exclude_layers.split(',') if l.strip()]

    remove_qdq_for_excluded_layers(args.onnx_path, args.output_path, exclude_layers)


if __name__ == '__main__':
    main()
